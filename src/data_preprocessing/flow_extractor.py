"""Flow extraction utilities for SD-MKD preprocessing.

This module reads traffic sources (PCAP/CSV/binary stream dumps), builds
bidirectional flow groupings using five-tuples, and emits dictionaries mapping
flow keys to ordered packets. The implementation favors clarity and extensible
hooks rather than raw speed, making it suitable for research-grade preprocessing
pipelines.
"""
from __future__ import annotations

import csv
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

try:  # Optional dependencies for PCAP parsing
    import dpkt
except Exception:  # pragma: no cover - optional
    dpkt = None

try:  # Optional
    from scapy.all import PcapReader  # type: ignore
except Exception:  # pragma: no cover - optional
    PcapReader = None

from .utils import FiveTuple, FlowPacket, ensure_bytes, five_tuple_from_tuple, get_logger


@dataclass
class FlowRecord:
    """Aggregated metadata for a single flow."""

    packets: List[FlowPacket]
    start_ts: float
    end_ts: float

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


class FlowExtractor:
    """Extract flows from PCAP/CSV/binary sources with timeout handling."""

    def __init__(self, timeout: float = 120.0, normalize_direction: bool = True, logger=None):
        self.timeout = timeout
        self.normalize_direction = normalize_direction
        self.logger = logger or get_logger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, path: str) -> Dict[str, FlowRecord]:
        """Dispatch to the appropriate parser based on file extension."""

        ext = os.path.splitext(path)[1].lower()
        if ext in {".pcap", ".pcapng"}:
            return self._parse_pcap(path)
        if ext == ".csv":
            return self._parse_csv(path)
        return self._parse_binary(path)

    # ------------------------------------------------------------------
    # PCAP parsing
    # ------------------------------------------------------------------
    def _parse_pcap(self, path: str) -> Dict[str, FlowRecord]:
        if PcapReader is None and dpkt is None:
            raise ImportError("PCAP parsing requires scapy or dpkt. Please install one of them.")

        self.logger.info(f"Parsing PCAP: {path}")
        flows: Dict[str, FlowRecord] = {}

        if PcapReader is not None:  # Prefer scapy for simplicity
            with PcapReader(path) as reader:  # type: ignore
                for pkt in reader:  # pragma: no cover - depends on scapy runtime
                    ts = float(pkt.time)
                    if not hasattr(pkt, "payload"):
                        continue
                    try:
                        ip = pkt[1]
                        l4 = ip.payload
                        proto = "TCP" if l4.name.lower() == "tcp" else "UDP"
                        five = FiveTuple(
                            src_ip=str(ip.src),
                            dst_ip=str(ip.dst),
                            src_port=int(getattr(l4, "sport", 0)),
                            dst_port=int(getattr(l4, "dport", 0)),
                            proto=proto,
                        )
                        payload = bytes(l4.payload)
                    except Exception:
                        continue
                    self._add_packet(flows, five, ts, payload)
        else:  # dpkt fallback
            with open(path, "rb") as f:  # pragma: no cover - depends on dpkt runtime
                pcap = dpkt.pcap.Reader(f)
                for ts, buf in pcap:
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if not isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                            continue
                        ip = eth.data
                        proto = "TCP" if isinstance(ip.data, dpkt.tcp.TCP) else "UDP"
                        l4 = ip.data
                        payload = bytes(l4.data)
                        five = FiveTuple(
                            src_ip=ip.src.hex(),
                            dst_ip=ip.dst.hex(),
                            src_port=int(getattr(l4, "sport", 0)),
                            dst_port=int(getattr(l4, "dport", 0)),
                            proto=proto,
                        )
                    except Exception:
                        continue
                    self._add_packet(flows, five, float(ts), payload)

        self._finalize(flows)
        return flows

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------
    def _parse_csv(self, path: str) -> Dict[str, FlowRecord]:
        """Parse CSV rows containing per-packet details.

        Expected columns (minimal): timestamp, src_ip, dst_ip, src_port, dst_port,
        protocol, payload_hex. Additional columns are ignored.
        """

        self.logger.info(f"Parsing CSV: {path}")
        flows: Dict[str, FlowRecord] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row.get("timestamp", 0))
                    five = FiveTuple(
                        src_ip=row["src_ip"],
                        dst_ip=row["dst_ip"],
                        src_port=int(row["src_port"]),
                        dst_port=int(row["dst_port"]),
                        proto=row.get("protocol", "TCP").upper(),
                    )
                    payload_hex = row.get("payload_hex", "")
                    payload = bytes.fromhex(payload_hex) if payload_hex else b""
                except Exception:
                    continue
                self._add_packet(flows, five, ts, payload)

        self._finalize(flows)
        return flows

    # ------------------------------------------------------------------
    # Binary parsing
    # ------------------------------------------------------------------
    def _parse_binary(self, path: str) -> Dict[str, FlowRecord]:
        """Parse a simple binary stream dump organized as length-prefixed packets.

        The expected format is: ``[five-tuple length][five-tuple utf-8 string][ts][payload length][payload]``
        repeated until EOF. This is intentionally flexible; adjust to your own
        format by editing this method.
        """

        self.logger.info(f"Parsing binary dump: {path}")
        flows: Dict[str, FlowRecord] = {}
        with open(path, "rb") as f:
            while True:
                len_bytes = f.read(2)
                if not len_bytes:
                    break
                (key_len,) = struct.unpack("!H", len_bytes)
                key = f.read(key_len).decode()
                ts_bytes = f.read(8)
                if len(ts_bytes) < 8:
                    break
                (ts,) = struct.unpack("!d", ts_bytes)
                payload_len_bytes = f.read(4)
                if len(payload_len_bytes) < 4:
                    break
                (p_len,) = struct.unpack("!I", payload_len_bytes)
                payload = f.read(p_len)

                try:
                    src_ip, rest = key.split("->")
                    dst_part, proto = rest.rsplit(":", 1)
                    dst_ip_port = dst_part.split(":")
                    dst_ip = dst_ip_port[0]
                    src_ip_addr, src_port_str = src_ip.split(":")
                    dst_port = int(dst_ip_port[1])
                    five = FiveTuple(src_ip=src_ip_addr, dst_ip=dst_ip, src_port=int(src_port_str), dst_port=dst_port, proto=proto)
                except ValueError:
                    continue

                self._add_packet(flows, five, ts, payload)

        self._finalize(flows)
        return flows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _add_packet(self, flows: Dict[str, FlowRecord], five: FiveTuple, ts: float, payload: bytes) -> None:
        key = five.canonical().to_key() if self.normalize_direction else five.to_key()
        pkt = FlowPacket(timestamp=ts, payload=ensure_bytes(payload), five_tuple=five)
        if key not in flows:
            flows[key] = FlowRecord(packets=[pkt], start_ts=ts, end_ts=ts)
            return

        record = flows[key]
        if ts - record.end_ts > self.timeout:
            # Timeout: start new flow instance with suffix key
            suffix = f"{key}#ts{int(ts)}"
            flows[suffix] = FlowRecord(packets=[pkt], start_ts=ts, end_ts=ts)
            return

        record.packets.append(pkt)
        record.end_ts = ts

    def _finalize(self, flows: Dict[str, FlowRecord]) -> None:
        for key, record in flows.items():
            record.packets.sort(key=lambda p: p.timestamp)
        self.logger.info(f"Collected {len(flows)} flows")


__all__ = ["FlowExtractor", "FlowRecord"]
