# # Importing Libraries
# import serial
# import time
# arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)
# def write_read(x):
#     arduino.write(bytes(x, 'utf-8'))
#     time.sleep(0.05)
#     data = arduino.readline()
#     return data
# while True:
#     num = input("Enter a number: ") 
#     value = write_read(num)
#     print(type(value))
import serial
import syslog
import time
port = '/dev/ttyACM0'
ard = serial.Serial(port,9600,timeout=5)
time.sleep(2)

i = 0

# while (i < 4):
#     # Serial write section

#     koordinat_x =  63
#     koordinat_y = 49
#     ard.flush()
#     setTemp1 = (koordinat_x)
#     setTemp2 = (koordinat_y)
#     print ("Python value sent: ")
#     print (setTemp1)
#     ard.write(setTemp1)
#     print (setTemp2)
#     ard.write(setTemp2)
#     time.sleep(1) 
#     # msg = ard.read(ard.inWaiting()) # read all characters in buffer
#     # print ("Message from arduino: ")
#     # print (msg)
#     i = i + 1
# else:
#     print ("Exiting")
# exit()

import struct
def kirim_koordinat(x,y):
    # string = b''
    
    string = struct.pack('i',x)
    print ("Python value sent: ",string)
    ard.write((string))

    string = struct.pack('i',y)
    print ("Python value sent: ",string)
    ard.write((string))

# Now send the string to the serial port
kirim_koordinat(256,255)
