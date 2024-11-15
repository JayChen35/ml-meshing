import torch
import platform

def is_plugged_in():
    if platform.system() == 'Windows':
        import ctypes
        SYSTEM_POWER_STATUS = ctypes.Structure([
            ('ACLineStatus', ctypes.c_byte),
            ('BatteryFlag', ctypes.c_byte),
            ('BatteryLifePercent', ctypes.c_byte),
            ('Reserved1', ctypes.c_byte),
            ('BatteryLifeTime', ctypes.c_ulong),
            ('BatteryFullLifeTime', ctypes.c_ulong),
        ])
        SYSTEM_POWER_STATUS_P = ctypes.POINTER(SYSTEM_POWER_STATUS)
        GetSystemPowerStatus = ctypes.cdll.kernel32.GetSystemPowerStatus
        status = SYSTEM_POWER_STATUS()
        if GetSystemPowerStatus(ctypes.byref(status)):
            return status.ACLineStatus == 1
        else:
            raise Exception("Could not determine power status")
    elif platform.system() == 'Darwin' or platform.system() == 'Linux':
        try:
            import psutil
            battery = psutil.sensors_battery()
            return battery.power_plugged
        except ImportError:
            raise Exception("psutil module is required on macOS/Linux to determine power status")
    else:
        raise Exception("Unsupported platform")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() and is_plugged_in() else "cpu")

# Example: Moving a tensor to the chosen device
example_tensor = torch.randn((3, 3)).to(device)
print(f"Device in use: {device}")