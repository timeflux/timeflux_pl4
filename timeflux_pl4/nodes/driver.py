from math import ceil
from struct import pack, unpack
import numpy as np
import ftd2xx as ftd
from timeflux.core.node import Node
from timeflux.helpers.clock import now
from timeflux.core.exceptions import WorkerInterrupt, TimefluxException

VID = 0x24f4        # Vendor ID
PID = 0x1000        # Product ID
HEADER = 0xAA       # Message header
ACK = 0x0000        # Acknowledgement
INFO = 0x0003       # Device info
START = 0x000B      # Start acquisition
STOP = 0x000C       # Stop acquisition
BUFFER_SIZE = 65536 # Input and output buffer size in bytes
PACKET_SIZE = 37    # Packet size in bytes


class InvalidChecksumException(TimefluxException):
    """Exception thrown when a PL4 packet cannot be parsed"""
    pass


class PhysioLOGX(Node):

    """
    Driver for the Mind Media PhysioLOG-4 (PL4) device.

    This node provides two streams. The first one (channels 1 and 2 at 1024 Hz) is
    expressed in uV. The second one (channels 3 and 4 at 256 Hz) is expressed in mV.

    In theory, we should be able to access the device via a serial interface using the
    FTDI VCP driver. The device is recognized, but does not appear in the /dev
    directory because the product and vendor IDs are unknown (at least on MacOS).
    Manually adding the IDs to the known devices table would require re-signing the
    driver with a kext-signing certificate. Instead, we install the D2XX driver, which
    allows synchronous access through a library and the Python ft2xx wrapper.

    Attributes:
        o_1024hz (Port): Channels 1 and 2, provides DataFrame.
        o_256hz (Port): Channels 3 and 4, provide DataFrame.

    Example:
        .. literalinclude:: /../../timeflux_pl4/test/graphs/pl4.yaml
           :language: yaml

    See
        - https://www.ftdichip.com/Products/ICs/FT232R.htm
        - https://www.ftdichip.com/Drivers/VCP.htm
        - https://www.ftdichip.com/Drivers/D2XX.htm
        - https://pypi.org/project/ftd2xx

    """

    def __init__(self):

        self.device = None

        # Setup
        try:
            # On Unix systems, we need to manually set the product and vendor IDs
            ftd.setVIDPID(VID, PID)
        except AttributeError:
            # The method is not available on Windows
            pass

        # Connect
        try:
            # Open the first FTDI device
            self.device = ftd.open(0)
            # Get info
            self.logger.info(self.device.getDeviceInfo())
        except ftd.ftd2xx.DeviceError:
            # Could not open device
            raise WorkerInterrupt('Could not open device')

        # Initialize connection
        if self.device:
            self.device.setBaudRate(921600)
            self.device.setFlowControl(ftd.defines.FLOW_NONE, 0, 0)
            self.device.setDataCharacteristics(ftd.defines.BITS_8, ftd.defines.STOP_BITS_1, ftd.defines.PARITY_NONE)
            self.device.setTimeouts(2000, 2000)
            self.device.setLatencyTimer(2)
            self.device.setUSBParameters(BUFFER_SIZE, BUFFER_SIZE)

        # Start acquisition
        self.packet_count = 0
        self.time_delta = {
            '1024Hz': np.timedelta64(int(1e9 / 1024), 'ns'),
            '256Hz': np.timedelta64(int(1e9 / 256), 'ns'),
        }
        self.start()
        self.time_start = now()


    def update(self):
        # How many bytes are available?
        queue = self.device.getQueueStatus()
        if queue == BUFFER_SIZE:
            self.logger.warn('The buffer is full. Please increase the graph rate.')
        # Prepare data containers
        # The device outputs channels 1-2 at 1024 Hz and channels 3-4 at 256 Hz.
        # One packet contains 4 samples for channels 1-2 and 1 sample for channels 3-4.
        data = {
            '1024Hz': {
                'data': { 'counter': [], '1': [], '2': [] },
                'index': []
            },
            '256Hz': {
                'data': { 'counter': [], '3': [], '4': [] },
                'index': []
            }
        }
        # Parse full packets
        for i in range(ceil(queue / PACKET_SIZE)):
            # Read one packet
            packet = self.read()
            if packet:
                try:
                    # Parse the packet
                    counter, samples = self.parse(packet)
                    # Append to the data container
                    data['1024Hz']['data']['counter'] += [counter] * 4
                    data['1024Hz']['data']['1'] += samples['1']
                    data['1024Hz']['data']['2'] += samples['2']
                    data['256Hz']['data']['counter'].append(counter)
                    data['256Hz']['data']['3'] += samples['3']
                    data['256Hz']['data']['4'] += samples['4']
                    # Infer timestamps from packet count and sample rate
                    # This will fail dramatically if too much packets are lost
                    self.packet_count += 1
                    start = self.time_start + self.time_delta['256Hz'] * self.packet_count
                    stop = start + self.time_delta['256Hz']
                    data['256Hz']['index'].append(start)
                    start = self.time_start + self.time_delta['1024Hz'] * self.packet_count * 4
                    stop = start + self.time_delta['1024Hz'] * 4
                    timestamps = list(np.arange(start, stop, self.time_delta['1024Hz']))
                    data['1024Hz']['index'] += timestamps
                except InvalidChecksumException:
                    pass
        # Output
        if len(data['1024Hz']['index']) > 0:
            self.o_1024hz.set(data['1024Hz']['data'], data['1024Hz']['index'])
            self.o_256hz.set(data['256Hz']['data'], data['256Hz']['index'])


    def terminate(self):
        self.stop()


    def version(self):
        self.command(INFO)
        # Header (2B) + Response ID (2B) + Response Size (2B) + Payload Size (10B) + Checksum (2B)
        data = self.device.read(18)
        #self.logger.debug(f'< {data}')
        version = unpack('>BBHHHHHLH', data)
        return {
            'device_id': version[4],
            'software_version': version[5],
            'hardware_version': version[6],
            'serial_number': version[7]
        }

    def start(self):
        self.device.purge()
        self.command(START)
        self.ack()

    def stop(self):
        self.command(STOP)
        self.device.purge()
        self.device.close()

    def command(self, command_id, payload=None):
        # Header (2B) + Command ID (2B) + Command Size (2B) + Payload Size + Checksum (2B)
        size = 8
        data = pack('>BBHH', HEADER, HEADER, command_id, size)
        checksum = 65536
        for byte in data:
            checksum -= byte
        data += pack('>H', checksum)
        #self.logger.debug(f'> {data}')
        self.device.write(data)

    def ack(self):
        # Header (2B) + Response ID (2B) + Response Size (2B) + Payload Size (41B) + Checksum (2B)
        size = 49
        data = self.device.read(size)
        #self.logger.debug(f'< {data}')
        if len(data) == size and int.from_bytes(data[2:4], byteorder='big') == ACK and data[6] == 0x00:
            return True
        return False

    def read(self):
        # A full packet is 37 bytes
        data = self.device.read(PACKET_SIZE)
        # Check if the packet starts with a header byte
        if data[0] == HEADER:
            return data
        # Oh snap! The packet is corrupted...
        # Look for the next header byte
        self.logger.warn('Invalid header')
        for index, byte in enumerate(data):
            if byte == HEADER:
                data = data[index:] + self.device.read(index)
                return data
        # Ahem... No luck
        return False

    def parse(self, data):
        # self.logger.debug(f'< {data}')
        # Validate checksum
        checksum = 0
        for byte in data:
            checksum += byte
        if (checksum % 256) != 0:
            self.logger.warn('Invalid checksum')
            raise InvalidChecksumException
        # Counter
        counter = data[1]
        # Samples
        samples = { '1': [], '2': [], '3': [], '4': [] }
        channels = ['1', '2', '3', '1', '2', '1', '2', '4', '1', '2']
        # Channels 1 and 2 are expressed in uV
        # LSB value ADC for channels 1 and 2: (((Vref * 2) / (resolution ADS1254)) / (gain of ina)) = (((2.048 * 2) / (2^24)) / 20.61161164) = 0.01184481006
        # Channels 3 and 4 are expressed in mV
        # LSB value ADC for channels 3 and 4: ((Vref * 2) / (resolution ADS1254)) / 1000 = ((2.048 * 2) / (2^24)) / 1000 = 0.000244140625
        adc = { '1': -0.01184481006, '2':  -0.01184481006, '3': -0.000244140625, '4': -0.000244140625 }
        for index, channel in enumerate(channels):
            start = 2 * index + index + 2
            stop = start + 3
            # Each sample is 3 bytes (2's complement)
            # We multiply the signed integer by the corresponding ADC to obtain the final value
            samples[channel].append(
                int.from_bytes(data[start:stop], byteorder='big', signed=True) * adc[channel]
            )
        return counter, samples
