import serial
import serial.tools.list_ports
import threading
import time
from datetime import datetime
import sys
import os
import msvcrt  # Windows专用，用于键盘输入检测

try:
    import keyboard  # 第三方库，需要安装：pip install keyboard

    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("警告：未安装keyboard库，部分快捷键功能可能不可用。")
    print("请使用: pip install keyboard")


class RS485Communicator:

    def __init__(self, preset_commands):
        self.serial_port = None
        self.connected = False
        self.running = False
        self.lock = threading.Lock()
        self.preset_commands = preset_commands
        self.keyboard_listening = False
        self.keyboard_thread = None

    def list_ports(self):
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect(self, port, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=1):
        try:
            if self.connected:
                self.disconnect()

            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=timeout
            )

            if self.serial_port.is_open:
                self.connected = True
                self.running = True
                # 启动接收线程
                threading.Thread(target=self._receive, daemon=True).start()
                # 启动键盘监听线程
                self._start_keyboard_listener()
                print(f"已连接到 {port}，参数：{baudrate},{bytesize},{parity},{stopbits}")
                print('-' * 100)
                return True
            return False
        except Exception as e:
            print(f"连接失败：{e}")
            return False

    def disconnect(self):
        self.running = False
        self._stop_keyboard_listener()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.connected = False
        print("已断开连接")

    def _receive(self):
        while self.running and self.connected:
            try:
                if self.serial_port.in_waiting > 0:
                    with self.lock:
                        data = self.serial_port.read(self.serial_port.in_waiting)

                    hex_str = ' '.join(f"{b:02X}" for b in data)
                    print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 收到数据：")
                    print(f"  十六进制：{hex_str}")
                time.sleep(0.01)
            except Exception as e:
                if self.running:
                    print(f"\n接收错误：{e}")
                break

    def _start_keyboard_listener(self):
        """启动键盘监听线程"""
        if not HAS_KEYBOARD:
            return

        def keyboard_listener():
            """键盘监听线程函数"""
            print("\n键盘快捷键已启用：")
            print("  F1-F12: 发送对应组号的指令")
            print("  ESC: 退出程序")
            print("  H: 显示帮助")
            print("  C: 清除屏幕")

            while self.keyboard_listening:
                try:
                    # 监听功能键 F1-F12
                    for i in range(1, 13):  # F1到F12
                        key_name = f'f{i}'
                        if keyboard.is_pressed(key_name):
                            if i <= len(self.preset_commands):
                                print(f"\n[快捷键] 按下F{i}，发送第{i}组指令")
                                self.send_selected(i - 1)  # 转换为0-based索引
                                time.sleep(0.3)  # 防抖延迟
                            else:
                                print(f"\n[快捷键] 第{i}组指令未定义")

                    # 监听其他功能键
                    if keyboard.is_pressed('esc'):
                        print("\n[快捷键] 按下ESC，退出程序")
                        self.running = False
                        os._exit(0)  # 强制退出

                    if keyboard.is_pressed('h'):
                        print("\n[快捷键] 按下H，显示帮助")
                        self._show_help()
                        time.sleep(0.3)

                    if keyboard.is_pressed('c'):
                        print("\n[快捷键] 按下C，清屏")
                        os.system('cls' if os.name == 'nt' else 'clear')
                        time.sleep(0.3)

                except Exception as e:
                    if self.keyboard_listening:
                        print(f"\n键盘监听错误：{e}")

                time.sleep(0.01)

        self.keyboard_listening = True
        self.keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        self.keyboard_thread.start()

    def _stop_keyboard_listener(self):
        """停止键盘监听"""
        self.keyboard_listening = False
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=0.5)

    def _show_help(self):
        """显示帮助信息"""
        print("\n" + "=" * 50)
        print("帮助信息")
        print("=" * 50)
        print("键盘快捷键：")
        print("  F1-F12: 发送对应组号的指令")
        print("  ESC: 退出程序")
        print("  H: 显示此帮助信息")
        print("  C: 清除屏幕")
        print("\n预设指令组：")
        for i, (key, commands) in enumerate(self.preset_commands.items()):
            print(f"  组{key}: {len(commands)}条指令")
            for j, cmd in enumerate(commands):
                if j == 0:
                    print(f"    第一条: {cmd[:50]}{'...' if len(cmd) > 50 else ''}")
        print("=" * 50)

    def send_selected(self, index):
        if not self.connected:
            print("未连接，先连接串口")
            return False

        if not (0 <= index < len(self.preset_commands)):
            print("无效的指令组号")
            return False

        command = self.preset_commands[str(index + 1)]  # 转换为1-based字符串键
        try:
            # 发送该组的所有指令
            for cmd_idx, cmd in enumerate(command):
                cmd_clean = cmd.replace(' ', '')
                if len(cmd_clean) % 2 != 0:
                    print(f"第{index + 1}组第{cmd_idx + 1}条指令长度必须为偶数")
                    continue

                data = bytes.fromhex(cmd_clean)
                crc = self._crc16(data)
                data_with_crc = data + crc

                with self.lock:
                    self.serial_port.write(data_with_crc)

                hex_full = ' '.join(f"{b:02X}" for b in data_with_crc)
                print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 发送第{index + 1}组第{cmd_idx + 1}条指令：")
                print(f"  完整指令：{hex_full}")

                # 如果是组内的多条指令，给点间隔
                if cmd_idx < len(command) - 1:
                    time.sleep(0.1)

            print("指令组发送完成")
            print('-' * 100)
            time.sleep(0.1)
            return True
        except Exception as e:
            print(f"发送错误：{e}")
            print('-' * 100)
            return False

    def _crc16(self, data):
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 0x0001 else crc >> 1
        return crc.to_bytes(2, byteorder='little')


def main():
    # 预设指令组配置y
    # ====================  RS485 RTU  ====================
    #
    # 可用串口：
    #   1. COM10
    #   2. COM4
    #   3. COM5
    # 选择串口（1-3）：1
    # 已连接到 COM10，参数：115200,8,N,1
    # ----------------------------------------------------------------------------------------------------
    #
    # 操作说明：输入发送对应组号，输入exit退出
    # 输入指令组号发送，输入exit退出：2
    # 程序错误：1
    PRESET_COMMANDS = {
        "1": [
            "27 10 00 0E 00 07 0E 00 FF 00 FF 00 FF 00 FF 00 FF 00 FF 00 FF"
        ],#快速-1
        "2": [
            "27 10 00 0E 00 07 0E 00 32 00 32 00 32 00 32 00 32 00 32 00 32"
        ],#慢速-2
        "3": [
            "27 10 00 07 00 07 0E 00 01 00 01 00 01 00 01 00 01 00 01 00 01",
            "27 10 00 23 00 07 0E 00 01 00 01 00 01 00 01 00 01 00 01 00 01"
        ],#小力矩-3
        "4": [
            "27 10 00 07 00 07 0E 00 FF 00 FF 00 FF 00 FF 00 FF 00 FF 00 FF",
        ],#大力矩-4
        "5": [
            "27 10 00 00 00 07 0E 00 FF 00 00 00 FE 00 FE 00 FE 00 FE 00 64",
        ],#伸直-5
        "6": [
            "27 10 00 00 00 07 0E 00 B4 00 00 00 A5 00 91 00 A5 00 A0 00 64",
        ],#抓紧-6
        "7": [
            "27 10 00 00 00 07 0E 00 00 00 00 00 B4 00 B4 00 B4 00 B4 00 64",
        ],
        "8": [
            "27 10 00 00 00 07 0E 00 C8 00 00 00 FF 00 00 00 00 00 00 00 64",
        ],
        "9": [
            "27 10 00 00 00 07 0E 00 00 00 00 00 FF 00 00 00 00 00 00 00 64",
        ],
        "10": [
            "27 10 00 00 00 07 0E 00 00 00 00 00 8E 00 00 00 00 00 00 00 64",
        ],
        "11": [
            "27 10 00 00 00 07 0E 00 00 00 00 00 FF 00 00 00 00 00 00 00 64",
        ],
        "12": [
            "27 04 00 29 00 00",
        ]
    }
    # 可以根据需要继续添加更多组，最多到F12（12组）

    comm = RS485Communicator(PRESET_COMMANDS)

    try:
        print("=" * 20, " RS485 RTU 通信测试工具 ", "=" * 20)
        print("版本：支持键盘快捷键 (F1-F12发送，ESC退出)")
        print("=" * 60)

        # 显示当前预设指令组
        print("\n预设指令组配置：")
        for key, commands in PRESET_COMMANDS.items():
            print(f"  组{key} ({len(commands)}条指令): {commands[0][:40]}...")

        ports = comm.list_ports()
        if not ports:
            print("\n无可用串口")
            return

        print("\n可用串口：")
        for i, port in enumerate(ports):
            print(f"  {i + 1}. {port}")

        while True:
            try:
                idx = int(input(f"\n选择串口 (1-{len(ports)})：").strip())
                if 1 <= idx <= len(ports):
                    break
                print(f"无效选择，请输入1-{len(ports)}之间的数字")
            except ValueError:
                print("请输入有效数字")

        # 串口参数（可根据需要修改）
        baudrate = 115200
        bytesize = 8
        parity = 'N'
        stopbits = 1

        if comm.connect(ports[idx - 1], baudrate, bytesize, parity, stopbits):
            if HAS_KEYBOARD:
                print("\n键盘快捷键已激活！")
                print("  按下 F1 发送第1组指令")
                print("  按下 F2 发送第2组指令")
                print("  ... 依此类推")
                print("  按下 ESC 退出程序")
                print("  按下 H 显示帮助")
                print("  按下 C 清除屏幕")
            else:
                print("\n注意：键盘快捷键功能未启用，请安装keyboard库")
                print("安装命令: pip install keyboard")

            print("\n或者，您也可以使用传统方式：")
            print("  输入指令组号发送")
            print("  输入exit退出")

            while comm.connected:
                try:
                    # 非阻塞键盘输入检测（备用方案，用于没有keyboard库的情况）
                    if not HAS_KEYBOARD and sys.platform == 'win32':
                        if msvcrt.kbhit():
                            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                            if key == '\x1b':  # ESC
                                print("\n按下ESC，退出程序")
                                break
                            elif key in ['1', '2', '3', '4', '5', '6']:
                                group_num = int(key)
                                print(f"\n[备用快捷键] 发送第{group_num}组指令")
                                comm.send_selected(group_num - 1)

                    # 传统输入方式
                    user_input = input("\n输入指令组号发送，输入help显示帮助，输入exit退出：").strip().lower()

                    if user_input == 'exit':
                        break
                    elif user_input == 'help':
                        comm._show_help()
                    elif user_input.isdigit():
                        group_num = int(user_input)
                        if 1 <= group_num <= len(PRESET_COMMANDS):
                            comm.send_selected(group_num - 1)
                        else:
                            print(f"无效的组号，请输入1-{len(PRESET_COMMANDS)}之间的数字")
                    else:
                        print("无效输入，请输入组号、help或exit")

                except KeyboardInterrupt:
                    print("\n\n检测到Ctrl+C，正在退出...")
                    break

        comm.disconnect()
    except Exception as e:
        print(f"程序错误：{e}")
        import traceback
        traceback.print_exc()
        comm.disconnect()
    finally:
        print("\n程序退出")
        time.sleep(1)


if __name__ == "__main__":
    # 检查必要的库
    if not HAS_KEYBOARD:
        print("提示：要使用完整的键盘快捷键功能，请安装keyboard库")
        print("安装命令: pip install keyboard")
        print("是否继续运行？(y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            sys.exit(0)

    main()