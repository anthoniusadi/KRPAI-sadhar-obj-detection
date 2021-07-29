import serial

serial1 = serial.Serial('/dev/ttyACM1', 9600)
def kirim(x,y):
    serial1.flush()
    x = str(x)
    y= str(y)
    x_Encode = x.encode()
    y_Encode = y.encode()

    x_Encode += b'\n'
    y_Encode += b'\n'
    serial1.write(x_Encode)
    serial1.write(y_Encode)
    
    print(x_Encode,y_Encode)
def get(message):
    pass
