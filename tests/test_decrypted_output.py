"""
Regression checks for the metadata MediaFlow emits alongside decrypted output.

Run directly, no test framework required:

    python tests/test_decrypted_output.py
"""

import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mediaflow_proxy.drm.decrypter import MP4Atom, MP4Decrypter, MP4Parser  # noqa: E402
from mediaflow_proxy.utils.mpd_utils import format_program_date_time  # noqa: E402


def box(atom_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + atom_type + payload


def full_box(atom_type: bytes, grouping_type: bytes) -> bytes:
    """A sample group box: 1 byte version, 3 byte flags, then the grouping type."""
    return box(atom_type, b"\x00\x00\x00\x00" + grouping_type)


def build_moof(data_offset: int) -> tuple[bytes, int]:
    """Builds a moof shaped like the ones a CENC live stream produces."""
    mfhd = box(b"mfhd", b"\x00\x00\x00\x00" + struct.pack(">I", 1))
    pssh = box(b"pssh", b"\x00" * 832)
    tfhd = box(b"tfhd", struct.pack(">II", 0, 1))
    # flags 0x000001 = data-offset-present, zero samples keeps the run trivial
    trun = box(b"trun", struct.pack(">IIi", 0x000001, 0, data_offset))
    senc = box(b"senc", b"\x00" * 16)
    saiz = box(b"saiz", b"\x00" * 12)
    saio = box(b"saio", b"\x00" * 12)
    seig_sbgp = full_box(b"sbgp", b"seig")
    seig_sgpd = full_box(b"sgpd", b"seig")
    roll_sbgp = full_box(b"sbgp", b"roll")

    traf = box(b"traf", tfhd + trun + senc + saiz + saio + seig_sbgp + seig_sgpd + roll_sbgp)
    moof = box(b"moof", mfhd + pssh + traf)

    dropped = len(pssh) + len(senc) + len(saiz) + len(saio) + len(seig_sbgp) + len(seig_sgpd)
    return moof, dropped


def child_types(atom: MP4Atom) -> list[bytes]:
    return [child.atom_type for child in MP4Parser(atom.data).list_atoms()]


def test_stale_protection_boxes_are_dropped():
    moof_bytes, _ = build_moof(data_offset=4000)
    moof = MP4Parser(memoryview(moof_bytes)).list_atoms()[0]

    processed = MP4Decrypter({})._process_moof(moof)

    assert b"pssh" not in child_types(processed), "pssh must not survive into decrypted output"
    traf = next(c for c in MP4Parser(processed.data).list_atoms() if c.atom_type == b"traf")
    kept = child_types(traf)
    assert b"senc" not in kept and b"saiz" not in kept and b"saio" not in kept
    assert b"sgpd" not in kept, "the seig sample group description must be dropped"
    # The roll sample group is unrelated to encryption and has to survive.
    assert kept.count(b"sbgp") == 1, f"exactly the roll sbgp should remain, got {kept}"
    assert bytes(next(c for c in MP4Parser(traf.data).list_atoms() if c.atom_type == b"sbgp").data[4:8]) == b"roll"


def test_trun_data_offset_absorbs_every_dropped_byte():
    """Every box removed ahead of mdat shifts it closer, so data_offset must shrink to match."""
    original_offset = 4000
    moof_bytes, dropped = build_moof(data_offset=original_offset)
    moof = MP4Parser(memoryview(moof_bytes)).list_atoms()[0]

    processed = MP4Decrypter({})._process_moof(moof)

    traf = next(c for c in MP4Parser(processed.data).list_atoms() if c.atom_type == b"traf")
    trun = next(c for c in MP4Parser(traf.data).list_atoms() if c.atom_type == b"trun")
    new_offset = struct.unpack_from(">i", trun.data, 8)[0]

    assert new_offset == original_offset - dropped, f"expected {original_offset - dropped}, got {new_offset}"
    # The shrunk moof and the corrected offset have to agree, or samples are read from the wrong place.
    assert len(moof_bytes) - processed.size == dropped


def test_program_date_time_has_a_single_timezone_designator():
    epoch = datetime.fromtimestamp(0, tz=timezone.utc)

    assert format_program_date_time(epoch) == "1970-01-01T00:00:00Z"
    assert format_program_date_time(epoch + timedelta(microseconds=360000)) == "1970-01-01T00:00:00.360000Z"
    # Naive values are treated as UTC.
    assert format_program_date_time(datetime(2026, 8, 14, 13, 33, 3)) == "2026-08-14T13:33:03Z"
    # A non UTC offset is normalised rather than left alongside a Z.
    aware = datetime(2026, 8, 14, 15, 33, 3, tzinfo=timezone(timedelta(hours=2)))
    assert format_program_date_time(aware) == "2026-08-14T13:33:03Z"

    for value in (epoch, aware, datetime(2026, 8, 14, 13, 33, 3)):
        formatted = format_program_date_time(value)
        assert formatted.count("Z") == 1 and "+" not in formatted, formatted


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"ok  {name}")
    print("all checks passed")
