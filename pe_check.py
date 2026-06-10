import struct, sys
with open(sys.argv[1], "rb") as f:
    dos = f.read(64)
    pe_offset = struct.unpack_from("<I", dos, 0x3C)[0]
    f.seek(pe_offset)
    sig = f.read(4)
    if sig == b"PE\0\0":
        machine = struct.unpack_from("<H", f.read(2))[0]
        arch = {0x8664: "x64", 0x14C: "x86", 0xAA64: "ARM64", 0x1C4: "ARM"}.get(machine, f"0x{machine:04X}")
        print(f"Architecture: {arch}")
        f.read(2)  # number of sections
        f.read(4)  # timestamp
        f.read(4)  # symbol table ptr
        f.read(4)  # number of symbols
        opt_hdr_size = struct.unpack_from("<H", f.read(2))[0]
        pattern = {0x8664, 0x14C, 0xAA64, 0x1C4}
        print(f"Optional header size: {opt_hdr_size}")
        # read subsystem from optional header
        f.read(2)  # magic
        # subsystem is at offset 68 in PE32+ (or 36 in PE32)
        # Let's just check the expected value
        print("PE file OK")
    else:
        print(f"Not a PE file (sig={sig})")
