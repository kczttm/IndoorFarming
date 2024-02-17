import serial

arduino = serial.Serial(port="./dev/cu.usbmodem11201 (Arduino Mega or Mega 2560)", baudrate=115200, timeout=.1)

def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

# Gripping Command
while True:
    # num = input("Enter a number: ") # Taking input from user
    value = write_read(1)
    if value == 999:
        print("Gripper has finished gripping and sensing.")
        break

print(value) # printing the value