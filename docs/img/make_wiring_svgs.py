"""Generate the Fritzing-style wiring diagrams in this folder.

Run:  python3 docs/img/make_wiring_svgs.py   →  wiring.svg (whole rig), wiring-blaster-stock.svg (trigger as found)
Edit the constants at the top when a measurement changes (e.g. which of black/blue is C).
Originally by Yuvraj (yuviji/act-athon, docs/02-hardware); diagram_rig() adds the USB side of the rig.
"""
from pathlib import Path

OUT = Path(__file__).parent

# ---- facts that may change after measuring (probe guide step 3) ----
C_WIRE = "blue"      # harness wire on microswitch pin 1 = C (pusher motor). Measured: 2-5 ohm to BAT- when pressed
NC_WIRE = "black"    # harness wire on microswitch pin 2 = NC (brake -> BAT-). Measured: ~0 ohm to BAT-
C_COLOR, NC_COLOR = "blue", "black"

WIRE = {  # fill, outline
    "red": ("#E53935", "#8E1B1B"), "black": ("#303030", "#000000"), "blue": ("#1E5BD8", "#0D2F7A"),
    "white": ("#FAFAFA", "#8A8A8A"), "yellow": ("#F9C80E", "#9A7A00"), "green": ("#2E9E44", "#175A25"),
    "grey": ("#A7ADB3", "#5F6368"), "orange": ("#FB8C00", "#9A5200"),
}

STYLE = """
<style>
  text{font-family:"Helvetica Neue",Helvetica,Arial,sans-serif}
  .title{font-size:20px;font-weight:700;fill:#1F2328}
  .sub{font-size:12px;fill:#57606A}
  .t{font-size:12px;fill:#1F2328}
  .ts{font-size:10.5px;fill:#57606A}
  .tb{font-size:12px;font-weight:700;fill:#1F2328}
  .pw{font-size:9px;fill:#FFFFFF;font-family:Menlo,Consolas,monospace}
  .pk{font-size:9px;fill:#1F2328;font-family:Menlo,Consolas,monospace}
  .pl{font-size:10px;fill:#1F2328;font-family:Menlo,Consolas,monospace;font-weight:700}
  .warn{font-size:11px;fill:#B42318;font-weight:700}
  .note-t{font-size:11px;fill:#3D3200}
</style>"""


def rounded(pts, r=12):
    """Polyline through pts with rounded corners, as an SVG path d string."""
    if len(pts) < 3:
        return "M" + " L".join(f"{x},{y}" for x, y in pts)
    d = [f"M{pts[0][0]},{pts[0][1]}"]
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        def toward(ax, ay, bx, by, dist):
            L = max(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5, 1e-6)
            k = min(dist, L / 2) / L
            return ax + (bx - ax) * k, ay + (by - ay) * k
        ax, ay = toward(x1, y1, x0, y0, r)
        bx, by = toward(x1, y1, x2, y2, r)
        d.append(f"L{ax:.1f},{ay:.1f} Q{x1},{y1} {bx:.1f},{by:.1f}")
    d.append(f"L{pts[-1][0]},{pts[-1][1]}")
    return " ".join(d)


def wire(pts, color, dash=False, ends=True):
    fill, edge = WIRE[color]
    d = rounded(pts)
    da = ' stroke-dasharray="10 7"' if dash else ""
    s = (f'<path d="{d}" fill="none" stroke="{edge}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"{da}/>'
         f'<path d="{d}" fill="none" stroke="{fill}" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"{da}/>')
    if ends:
        for x, y in (pts[0], pts[-1]):
            s += f'<circle cx="{x}" cy="{y}" r="3.2" fill="#C9CED4" stroke="#6B7178" stroke-width="1"/>'
    return s


def text(x, y, s, cls="t", anchor="start", rot=None):
    tr = f' transform="rotate({rot} {x} {y})"' if rot else ""
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{tr}>{s}</text>'


def note(x, y, w, lines, warn=False):
    h = 16 + 15 * len(lines)
    fill, edge = ("#FDE2E1", "#D92D20") if warn else ("#FFF6B8", "#D8B800")
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{edge}"/>'
    s += f'<path d="M{x+w-14},{y+h} L{x+w},{y+h-14} L{x+w},{y+h} Z" fill="{edge}" opacity=".35"/>'
    for i, ln in enumerate(lines):
        s += text(x + 8, y + 18 + 15 * i, ln, "note-t")
    return s


def canvas(w, h, title, sub):
    grid = (f'<defs><pattern id="g" width="20" height="20" patternUnits="userSpaceOnUse">'
            f'<path d="M20,0 L0,0 0,20" fill="none" stroke="#EEF1F4" stroke-width="1"/></pattern></defs>'
            f'<rect width="{w}" height="{h}" fill="#FFFFFF"/><rect width="{w}" height="{h}" fill="url(#g)"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">'
            + STYLE + grid + text(24, 36, title, "title") + text(24, 56, sub, "sub"))


# ------------------------------------------------------------------ parts (each returns svg, pins)

def battery_pack(x, y):
    s = f'<rect x="{x}" y="{y}" width="190" height="96" rx="6" fill="#26282B" stroke="#000"/>'
    for row in range(2):
        for i in range(4):
            cx, cy = x + 12 + i * 44, y + 12 + row * 40
            s += f'<rect x="{cx}" y="{cy}" width="36" height="30" rx="3" fill="#C98A1B" stroke="#6B4A0E"/>'
            s += f'<rect x="{cx+36}" y="{cy+10}" width="4" height="10" fill="#C9CED4"/>'
            s += text(cx + 18, cy + 19, "AA", "pk", "middle")
    s += text(x, y + 114, "Battery pack · 8× AA (~12 V)", "tb")
    s += text(x, y + 129, "count from product listing; verify", "ts")
    return s, {"+": (x + 190, y + 30), "-": (x + 190, y + 66)}


def jst(x, y):
    return f'<rect x="{x}" y="{y}" width="26" height="18" rx="2" fill="#1B1B1B" stroke="#000"/>' \
           f'<rect x="{x+4}" y="{y+4}" width="7" height="10" fill="#555"/><rect x="{x+15}" y="{y+4}" width="7" height="10" fill="#555"/>'


def junction_board(x, y, flip=False):
    w, h = 230, 150
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#3F8F3A" stroke="#1E4D1B" stroke-width="1.5"/>'
    s += f'<rect x="{x+4}" y="{y+4}" width="{w-8}" height="{h-8}" rx="4" fill="none" stroke="#6FBF5E" stroke-width="1" opacity=".6"/>'
    # F1
    s += f'<rect x="{x+80}" y="{y+62}" width="56" height="26" rx="2" fill="#1B1B1B"/>'
    s += f'<rect x="{x+76}" y="{y+62}" width="8" height="26" fill="#C9CED4"/><rect x="{x+132}" y="{y+62}" width="8" height="26" fill="#C9CED4"/>'
    s += text(x + 108, y + 80, "JK 300", "pw", "middle")
    s += text(x + 108, y + 102, "F1 · 3 A PTC fuse", "pw", "middle")
    s += text(x + 12, y + 50, "36918 PCBAX-003R", "pw")
    pins = {}
    pads = {  # name: (dx, dy)
        "A-": (30, 14), "A+": (52, 14), "B-": (170, 14), "B+": (192, 14),
        "BAT-": (30, 136), "BAT+": (52, 136),
        "H_red": (140, 136), "H_black": (162, 136), "H_blue": (184, 136), "H_white": (206, 136),
    }
    for n, (dx, dy) in pads.items():
        if flip:
            dy = h - dy
        s += f'<circle cx="{x+dx}" cy="{y+dy}" r="6" fill="#D7DBDF" stroke="#7C8288"/><circle cx="{x+dx}" cy="{y+dy}" r="2.4" fill="#5F6368"/>'
        pins[n] = (x + dx, y + dy)
    return s, pins


def motor(x, y, label, sub):
    s = f'<rect x="{x}" y="{y}" width="84" height="46" rx="8" fill="#C9CED4" stroke="#6B7178" stroke-width="1.5"/>'
    s += f'<rect x="{x+84}" y="{y+13}" width="12" height="20" rx="2" fill="#9AA0A6" stroke="#6B7178"/>'
    s += f'<rect x="{x-8}" y="{y+8}" width="8" height="30" rx="2" fill="#8B5A2B" stroke="#5A3A1A"/>'
    s += text(x + 42, y + 28, "M", "tb", "middle")
    s += text(x + 42, y - 22, label, "tb", "middle") + text(x + 42, y - 8, sub, "ts", "middle")
    return s, {"-": (x + 18, y + 46), "+": (x + 66, y + 46)}


def microswitch(x, y):
    s = ""
    pins = {}
    for i, n in enumerate("LMR"):
        px = x + 16 + i * 34
        s += f'<rect x="{px-4}" y="{y-18}" width="8" height="20" fill="#C9CED4" stroke="#6B7178"/>'
        pins[n] = (px, y - 18)
    s += f'<rect x="{x}" y="{y}" width="100" height="52" rx="3" fill="#1E1E1E" stroke="#000"/>'
    s += f'<circle cx="{x+28}" cy="{y+26}" r="6" fill="#333" stroke="#000"/><circle cx="{x+72}" cy="{y+26}" r="6" fill="#333" stroke="#000"/>'
    for i, n in enumerate("LMR"):
        s += text(x + 16 + i * 34, y + 46, n, "pw", "middle")
    s += f'<rect x="{x+44}" y="{y+52}" width="12" height="6" fill="#E53935"/>'
    s += f'<path d="M{x+4},{y+64} L{x+104},{y+58}" stroke="#C9CED4" stroke-width="3" stroke-linecap="round"/>'
    return s, pins


def blade_and_leaf(x, y):
    """Metal blade rising from pin M (x,y = pin top) and the white leaf ending next to it."""
    s = f'<rect x="{x+4}" y="{y-86}" width="7" height="88" fill="#C9CED4" stroke="#6B7178"/>'
    s += f'<path d="M{x+70},{y-98} L{x+36},{y-74} L{x+16},{y-66}" fill="none" stroke="#6B7178" stroke-width="5" stroke-linecap="round"/>'
    s += f'<path d="M{x+70},{y-98} L{x+36},{y-74} L{x+16},{y-66}" fill="none" stroke="#DDE1E5" stroke-width="3" stroke-linecap="round"/>'
    return s, {"leaf": (x + 70, y - 98), "blade_top": (x + 7, y - 86)}


def esp32(x, y):
    w, h = 124, 392
    left = ["EN", "VP", "VN", "D34", "D35", "D32", "D33", "D25", "D26", "D27", "D14", "D12", "D13", "GND", "VIN"]
    right = ["D23", "D22", "TX0", "RX0", "D21", "D19", "D18", "D5", "TX2", "RX2", "D4", "D2", "D15", "GND", "3V3"]
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#1C1C1E" stroke="#000"/>'
    s += f'<rect x="{x+18}" y="{y+10}" width="88" height="96" rx="3" fill="#B8BEC4" stroke="#6B7178"/>'
    s += f'<path d="M{x+26},{y+22} h72 M{x+26},{y+22} v10 h12 v-10 m12,0 v10 h12 v-10 m12,0 v10 h12 v-10" fill="none" stroke="#6B7178" stroke-width="1.5"/>'
    s += text(x + 62, y + 70, "ESP32-", "pk", "middle") + text(x + 62, y + 82, "WROOM-32", "pk", "middle")
    s += f'<rect x="{x+44}" y="{y+h-14}" width="36" height="18" rx="2" fill="#C9CED4" stroke="#6B7178"/>'
    s += text(x + 62, y + h - 22, "USB", "pw", "middle")
    pins = {}
    for i, n in enumerate(left):
        py = y + 124 + i * 17
        s += f'<rect x="{x+4}" y="{py-4}" width="8" height="8" fill="#2A2A2A"/><circle cx="{x+8}" cy="{py}" r="2.8" fill="#D4AF37"/>'
        s += text(x + 16, py + 3, n, "pw")
        pins["L_" + n] = (x + 8, py)
    for i, n in enumerate(right):
        py = y + 124 + i * 17
        s += f'<rect x="{x+w-12}" y="{py-4}" width="8" height="8" fill="#2A2A2A"/><circle cx="{x+w-8}" cy="{py}" r="2.8" fill="#D4AF37"/>'
        s += text(x + w - 16, py + 3, n, "pw", "end")
        pins["R_" + n] = (x + w - 8, py)
    pins["USB"] = (x + 62, y + h + 4)
    s += text(x + 62, y - 24, "ESP32-A · fire-ctl", "tb", "middle")
    s += text(x + 62, y - 10, "DOIT DevKit V1 (30-pin)", "ts", "middle")
    return s, pins


def relay_board(x, y):
    """Inland 8-ch relay board, drawn upright like the photo: header left, relays centre, screw terminals right."""
    w, h = 214, 540
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#1F5FAF" stroke="#0E3566" stroke-width="1.5"/>'
    s += text(x + 12, y + 20, "Inland 8-ch relay (5 V coils)", "pw")
    pins = {}
    hdr = ["VCC", "GND", "IN1", "IN2", "IN3", "IN4", "IN5", "IN6", "IN7", "IN8", "COM", "GND2"]
    hx, hy = x + 16, y + 190
    s += f'<rect x="{hx-7}" y="{hy-9}" width="14" height="{len(hdr)*14+4}" fill="#111"/>'
    for i, n in enumerate(hdr):
        py = hy + i * 14
        s += f'<rect x="{hx-3}" y="{py-3}" width="6" height="6" fill="#D4AF37"/>'
        s += text(hx + 12, py + 3, "GND" if n == "GND2" else n, "pw")
        pins[n] = (hx, py)
    # bridge on IN1..IN8 (back side)
    s += f'<path d="M{hx-10},{hy+28} V{hy+126}" stroke="#E0E3E6" stroke-width="2" stroke-dasharray="3 2"/>'
    for k in range(8):
        ry = y + 34 + k * 62
        s += f'<rect x="{x+86}" y="{ry}" width="78" height="54" rx="2" fill="#3C8DE0" stroke="#174E8C"/>'
        s += text(x + 125, ry + 24, f"K{k+1}", "pw", "middle")
        s += text(x + 125, ry + 38, "SRD-05VDC", "pw", "middle")
        s += f'<rect x="{x+172}" y="{ry}" width="34" height="54" fill="#2D7BD1" stroke="#174E8C"/>'
        for j in range(3):
            sy = ry + 9 + j * 18
            s += f'<circle cx="{x+189}" cy="{sy}" r="6.5" fill="#D7DBDF" stroke="#6B7178"/>'
            s += f'<path d="M{x+184},{sy} h10" stroke="#6B7178" stroke-width="1.5"/>'
            pins[f"K{k+1}_{j}"] = (x + 189, sy)
    return s, pins


def usb_brick(x, y):
    s = f'<rect x="{x}" y="{y}" width="96" height="60" rx="8" fill="#F4F5F7" stroke="#8A9096" stroke-width="1.5"/>'
    s += f'<rect x="{x+96}" y="{y+20}" width="10" height="20" fill="#C9CED4" stroke="#6B7178"/>'
    s += text(x + 48, y + 26, "5 V buck", "tb", "middle") + text(x + 48, y + 42, "HW-688, fed by", "ts", "middle") + text(x + 48, y + 54, "12 V adapter", "ts", "middle")
    return s, {"+": (x + 106, y + 24), "-": (x + 106, y + 36)}


def laptop(x, y):
    s = f'<rect x="{x}" y="{y}" width="120" height="76" rx="5" fill="#3A3F45" stroke="#1B1E21"/>'
    s += f'<rect x="{x+8}" y="{y+8}" width="104" height="60" rx="2" fill="#6E8BA8"/>'
    s += f'<rect x="{x-10}" y="{y+76}" width="140" height="10" rx="3" fill="#8A9096"/>'
    s += text(x + 60, y + 42, "rigd", "pw", "middle")
    s += text(x + 60, y - 8, "Pi 5 / laptop", "tb", "middle")
    return s, {"usb": (x + 130, y + 81)}


def gnd(x, y, lead=12):
    """Ground symbol hanging below (x, y). All grounds with this symbol are the logic ground."""
    s = f'<path d="M{x},{y} v{lead}" stroke="#000" stroke-width="2"/>'
    for i, half in enumerate((11, 7, 3)):
        yy = y + lead + i * 4
        s += f'<path d="M{x-half},{yy} h{2*half}" stroke="#000" stroke-width="2"/>'
    return s


def battery_compact(x, y):
    """Battery pack with its 2 leads on the bottom edge."""
    s = f'<rect x="{x}" y="{y}" width="112" height="70" rx="5" fill="#26282B" stroke="#000"/>'
    for i in range(4):
        s += f'<rect x="{x+8+i*25}" y="{y+8}" width="21" height="22" rx="2" fill="#C98A1B" stroke="#6B4A0E"/>'
    s += text(x + 56, y + 46, "8× AA · ~12 V", "pw", "middle")
    s += text(x + 56, y + 60, "battery pack", "pw", "middle")
    return s, {"-": (x + 40, y + 70), "+": (x + 62, y + 70)}


# ------------------------------------------------------------------ diagram 1: blaster as found

def diagram_stock():
    W, H = 1120, 700
    s = canvas(W, H, "Dart Storm · trigger wiring as found (stock)",
               "Everything meets on the junction board. There is no controller: the switches carry motor current directly.")
    jb, J = junction_board(380, 250)
    ma, MA = motor(300, 110, "Motor pair A", "to flywheel cage")
    mb, MB = motor(580, 110, "Motor pair B", "to cage (small green board)")
    bp, BP = battery_pack(40, 470)
    ms, MS = microswitch(640, 560)
    bl, BL = blade_and_leaf(*MS["M"])

    s += ma + mb + bp
    s += wire([MA["-"], (MA["-"][0], 200), (J["A-"][0], 200), J["A-"]], "black")
    s += wire([MA["+"], (MA["+"][0], 215), (J["A+"][0], 215), J["A+"]], "red")
    s += wire([MB["-"], (MB["-"][0], 200), (J["B-"][0], 200), J["B-"]], "black")
    s += wire([MB["+"], (MB["+"][0], 215), (J["B+"][0], 215), J["B+"]], "red")
    s += jb
    # battery through a 2-pin plug
    s += wire([BP["+"], (300, BP["+"][1]), (300, 440)], "red", ends=False)
    s += wire([BP["-"], (320, BP["-"][1]), (320, 440)], "black", ends=False)
    s += jst(297, 422)
    s += wire([(305, 422), (305, 405), (J["BAT+"][0], 420), J["BAT+"]], "red")
    s += wire([(315, 422), (315, 412), (J["BAT-"][0], 428), J["BAT-"]], "black")
    s += text(290, 470, "2-pin plug", "ts", "end")
    # harness
    s += bl
    s += wire([J["H_red"], (J["H_red"][0], 470), (MS["M"][0] - 20, 470), (MS["M"][0] - 20, 520), MS["M"]], "red")
    s += wire([J["H_black"], (J["H_black"][0], 490), (MS["L"][0], 490), MS["L"]], "black")
    s += wire([J["H_blue"], (J["H_blue"][0], 440), (MS["R"][0] + 40, 440), (MS["R"][0] + 40, 520), MS["R"]], "blue")
    s += wire([J["H_white"], (J["H_white"][0], 410), (BL["leaf"][0] + 40, 410), (BL["leaf"][0] + 40, BL["leaf"][1]), BL["leaf"]], "white")
    s += ms
    s += text(MS["L"][0] + 34, 648, "Microswitch (SPDT, 5 A)", "tb", "middle") + text(MS["L"][0] + 34, 663, "viewed from the back: markings read 1 3 2 from the front", "ts", "middle")

    # pin role labels
    s += text(MS["L"][0] - 8, 548, "2 · NC", "pl", "end")
    s += text(MS["M"][0] + 16, 528, "3 · NO", "pl")
    s += text(MS["R"][0] + 50, 548, "1 · C", "pl")

    s += note(40, 90, 230, ["Rev: half pull pushes the white", "leaf onto the metal blade on", "pin M (red) → flywheels spin."])
    s += note(40, 170, 230, ["Fire: full pull presses the", "microswitch → red joins C →", "pusher runs. Released: C–NC", "shorts the pusher = brake."])
    s += note(850, 250, 250, ["Measured + matches the switch marks:", "red = 3 · NO", "blue = 1 · C (2–5 Ω to BAT− when", "pressed: through the pusher motor)", "black = 2 · NC (≈ 0 Ω to BAT−, brake)"])
    s += note(850, 380, 250, ["F1 'JK 300' = resettable fuse,", "~3 A hold / ~6 A trip. Red harness", "wire is BAT+ after F1 (inferred).", "Which motor pair is the pusher: TBD."])
    return s + "</svg>"


# ------------------------------------------------------------------ diagram 2: fire path with relay (D-011: always rev)

def power_switch(x, y):
    """KCD1 rocker (SPST, 2 prongs) used as the blaster master switch. Prongs at the bottom."""
    s = f'<rect x="{x}" y="{y}" width="64" height="36" rx="4" fill="#1E1E1E" stroke="#000"/>'
    s += f'<path d="M{x+8},{y+8} L{x+56},{y+4} L{x+56},{y+30} L{x+8},{y+30} Z" fill="#D32F2F" stroke="#7F1D1D"/>'
    s += text(x + 20, y + 23, "I", "pw", "middle") + text(x + 44, y + 23, "O", "pw", "middle")
    pins = {}
    for n, dx in (("a", 18), ("b", 46)):
        s += f'<rect x="{x+dx-4}" y="{y+36}" width="8" height="14" fill="#C9CED4" stroke="#6B7178"/>'
        pins[n] = (x + dx, y + 50)
    s += text(x + 32, y - 30, "Blaster power + rev", "tb", "middle")
    s += text(x + 32, y - 16, "KCD1 rocker · 10 A", "ts", "middle")
    return s, pins


def logic_side(s, host=True):
    """Laptop (optional), ESP32, 5 V supply and relay board. Returns (svg, relay pins, ESP32 pins)."""
    e, E = esp32(250, 140)
    ub, UB = usb_brick(40, 640)
    rb, R = relay_board(620, 110)
    s += e + ub
    if host:
        lp, LP = laptop(40, 110)
        s += lp
        s += wire([LP["usb"], (200, LP["usb"][1]), (200, 566), (E["USB"][0], 566), E["USB"]], "grey")
        s += text(208, 330, "USB serial · 115200", "ts", rot=90)
    s += wire([E["L_D13"], (236, E["L_D13"][1]), (236, 96), (600, 96), (600, R["IN1"][1]), R["IN1"]], "yellow")
    s += text(420, 90, "GPIO13 → IN1 (HIGH = relay on)", "ts", "middle")
    s += wire([E["R_GND"], (400, E["R_GND"][1])], "black") + gnd(400, E["R_GND"][1], 10)
    s += wire([R["GND"], (560, R["GND"][1]), (560, 360)], "black") + gnd(560, 360, 4)
    s += wire([UB["+"], (178, UB["+"][1]), (178, 760), (580, 760), (580, R["VCC"][1]), R["VCC"]], "red")
    s += wire([UB["-"], (152, UB["-"][1]), (152, 720)], "black") + gnd(152, 720, 4)
    # operator button: GPIO25 -> button -> GND (acts on release; mode set by the laptop)
    y25 = E["L_D25"][1]
    s += wire([E["L_D25"], (146, y25)], "orange")
    s += f'<circle cx="120" cy="{y25}" r="26" fill="#2A2D31" stroke="#000"/><circle cx="120" cy="{y25}" r="16" fill="#D32F2F" stroke="#7F1D1D"/>'
    s += wire([(94, y25 + 8), (70, y25 + 8), (70, y25 + 22)], "black", ends=False) + gnd(70, y25 + 22, 2)
    s += text(110, y25 + 48, "Operator button (D25)", "tb", "middle")
    s += text(110, y25 + 62, "release: train = fire", "ts", "middle")
    s += text(110, y25 + 75, "test = turn done", "ts", "middle")
    s += rb
    s += note(24, 250, 166, ["⏚ = logic ground:", "ESP32, relay header,", "buck output −.", "Never the blaster side."])
    return s, R, E


def diagram_relay():
    W, H = 1300, 800
    s = canvas(W, H, "Fire path · ESP32-A → relay → blaster (D-010, D-011)",
               "Flywheels spin whenever the power switch is ON. The relay only drives the pusher. The two circuits share no wire.")
    s, R, _ = logic_side(s)
    com, nc, no = R["K3_0"], R["K3_1"], R["K3_2"]
    s += text(no[0] + 14, no[1] + 4, "NO", "pl") + text(com[0] + 14, com[1] + 4, "COM", "pl") + text(nc[0] + 14, nc[1] + 4, "NC", "pl")
    jb, J = junction_board(1000, 560, flip=True)
    bp, BP = battery_compact(900, 400)
    ps, PS = power_switch(1040, 420)
    s += jb + bp + ps
    s += text(1115, 736, "motor pairs (see stock diagram)", "ts", "middle")
    # battery - straight to BAT-; battery + through the power switch (C -> A) to BAT+
    s += wire([BP["-"], (BP["-"][0], 540), (J["BAT-"][0], 540), J["BAT-"]], "black")
    s += wire([BP["+"], (BP["+"][0], 500), (PS["a"][0], 500), PS["a"]], "red")
    s += wire([PS["b"], (PS["b"][0], 520), (J["BAT+"][0], 520), J["BAT+"]], "red")
    # harness -> relay; white joins red on NO (always rev)
    s += wire([J["H_red"], (J["H_red"][0], no[1]), no], "red")
    s += wire([J["H_black"], (J["H_black"][0], nc[1]), nc], NC_COLOR)
    s += wire([J["H_blue"], (J["H_blue"][0], com[1]), com], C_COLOR)
    s += wire([J["H_white"], (J["H_white"][0], 546), (1272, 546), (1272, 760), (860, 760), (860, no[1])], "white", ends=False)
    s += f'<circle cx="860" cy="{no[1]}" r="5.5" fill="#6B7178"/>'
    s += text(866, 350, "white joins red", "ts") + text(866, 364, "(same NO screw)", "ts")
    s += text(900, 96, "Harness → relay", "tb")
    s += text(900, 114, "red + white → NO (always rev)", "t")
    s += text(900, 131, f"{C_WIRE} (C, pusher motor) → COM", "t")
    s += text(900, 148, f"{NC_WIRE} (NC, brake) → NC", "t")
    s += note(900, 166, 330, ["Screw order COM / NC / NO here is a placeholder:", "use your labels. IN1–IN8 are bridged."])
    s += note(620, 662, 226, [f"{C_WIRE} = C → COM, {NC_WIRE} = NC → NC.", "Swapped = battery short."], warn=True)
    s += note(24, 470, 166, ["Rocker = blaster power", "+ rev: the only switch.", "OFF = nothing can fire.", "`rigctl mode train|test`", "picks the button's job."])
    return s + "</svg>"


# ------------------------------------------------------------------ diagram 2a: bench setup (what we wire now)

def diagram_bench():
    W, H = 1180, 780
    s = canvas(W, H, "Bench setup · relay test before the blaster (v0)",
               "5 V buck output straight to relay VCC, GPIO13 to IN1, blaster NOT connected.")
    s, R, _ = logic_side(s)
    com, nc, no = R["K5_0"], R["K5_1"], R["K5_2"]
    s += text(no[0] + 14, no[1] + 4, "NO", "pl") + text(com[0] + 14, com[1] + 4, "COM", "pl") + text(nc[0] + 14, nc[1] + 4, "NC", "pl")
    s += text(872, 424, "← nothing connected yet", "t")
    s += note(900, 140, 250, ["Test (rigd + rigctl):", "rigctl fire 150 → click", "fire twice fast → cooldown", "mode train: button release → click", "mode test: release → no click", "relay held on (IN1 → 3V3):", "NO beeps with COM only"])
    s += note(24, 470, 166, ["Bench only: unplug", "the 12 V adapter to", "cut relay power."])
    return s + "</svg>"


# ------------------------------------------------------------------ diagram 3: ESP32-A, v0 (only what firing needs)

def diagram_esp32_v0():
    W, H = 980, 600
    s = canvas(W, H, "ESP32-A · v0 wiring (MVP)",
               "Two wires: GPIO13 drives the relay, GPIO25 reads the operator button. ⏚ = ESP32 GND.")
    e, E = esp32(560, 110)
    s += e
    # optional operator fire button (needed once we record teleop data with fire labels)
    y25 = E["L_D25"][1]
    s += f'<circle cx="300" cy="{y25}" r="26" fill="#2A2D31" stroke="#000"/><circle cx="300" cy="{y25}" r="16" fill="#D32F2F" stroke="#7F1D1D"/>'
    s += wire([(326, y25), E["L_D25"]], "orange")
    s += wire([(274, y25 + 8), (250, y25 + 8), (250, y25 + 20)], "black", ends=False) + gnd(250, y25 + 20, 2)
    s += text(300, y25 - 50, "Operator button (acts on RELEASE)", "tb", "middle")
    s += text(300, y25 - 36, "train: fire · test: \"my turn is done\"", "ts", "middle")
    s += text(450, y25 - 6, "GPIO25", "ts", "middle")
    # GPIO13 -> relay IN1
    s += wire([E["L_D13"], (480, E["L_D13"][1]), (480, 540), (200, 540)], "yellow")
    s += text(190, 544, "to relay IN1 (fire-path diagram)", "t", "end")
    s += wire([E["L_GND"], (536, E["L_GND"][1])], "black", ends=False) + gnd(536, E["L_GND"][1], 8)
    # USB to host
    s += wire([E["USB"], (E["USB"][0], 560), (700, 560)], "grey")
    s += text(708, 564, "USB → Pi 5 / laptop (rigd)", "t")
    s += note(720, 110, 240, ["One button, two modes (D-012):", "rigctl mode train → release fires", "rigctl mode test → release only", "signals (tic-tac-toe turn done).", "Scripts set the mode at start."])
    s += note(720, 210, 240, ["MVP firmware (D-014) uses only", "GPIO13 + GPIO25. No LED, buzzer", "or laser: every other pin is free."])
    return s + "</svg>"

# ------------------------------------------------------------------ whole rig: USB side + fire path (this repo's wiring.svg)

def arm_icon(x, y, color="#F0A030"):
    s = f'<rect x="{x}" y="{y+34}" width="54" height="10" rx="2" fill="#4A4F55"/>'
    s += f'<path d="M{x+27},{y+34} L{x+18},{y+12} L{x+46},{y+2}" fill="none" stroke="{color}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>'
    s += f'<circle cx="{x+18}" cy="{y+12}" r="3.5" fill="#222"/><circle cx="{x+27}" cy="{y+34}" r="3.5" fill="#222"/>'
    return s


def camera_icon(x, y):
    s = f'<rect x="{x+6}" y="{y+6}" width="44" height="30" rx="5" fill="#2A2D31" stroke="#000"/>'
    s += f'<circle cx="{x+28}" cy="{y+21}" r="10" fill="#7FB2E5" stroke="#1B1E21" stroke-width="2"/><circle cx="{x+28}" cy="{y+21}" r="4" fill="#123A63"/>'
    return s


def diagram_rig():
    W, H, DX = 1580, 820, 280  # the fire path (logic + blaster side) is drawn shifted right by DX
    s = canvas(W, H, "Wiring · SO-101 dart blaster",
               "USB side and logic side share one ground; the blaster side shares no wire with them. "
               "Flywheels spin whenever the rocker is ON. The relay only drives the pusher.")
    # laptop + powered USB hub + devices (left column)
    s += f'<rect x="60" y="110" width="120" height="76" rx="5" fill="#3A3F45" stroke="#1B1E21"/>'
    s += f'<rect x="68" y="118" width="104" height="60" rx="2" fill="#6E8BA8"/><rect x="50" y="186" width="140" height="10" rx="3" fill="#8A9096"/>'
    s += text(120, 144, "Claude +", "pw", "middle") + text(120, 157, "pi0.7 client", "pw", "middle")
    s += text(120, 100, "Laptop", "tb", "middle")
    hx, hy, hh = 228, 250, 400
    s += wire([(190, 191), (243, 191), (243, hy)], "grey")
    devices = [  # (label, sub, icon, y)
        ("Follower arm", "SO-101 · moves the blaster", lambda x, y: arm_icon(x, y), 300),
        ("Leader arm", "teleop only (recording)", lambda x, y: arm_icon(x, y, "#9CC3E6"), 390),
        ("Front camera", "behind the blaster, sees board", camera_icon, 480),
        ("Wrist camera", "on the blaster, above barrels", camera_icon, 570),
    ]
    for label, sub_, icon, y in devices:
        s += wire([(86, y + 22), (hx, y + 22)], "grey")
        s += icon(30, y)
        s += text(30, y + 62, label, "tb") + text(30, y + 76, sub_, "ts")
    s += f'<rect x="{hx}" y="{hy}" width="30" height="{hh}" rx="5" fill="#45484C" stroke="#1B1E21"/>'
    for y in [322, 412, 502, 592, 622]:
        s += f'<rect x="{hx+4}" y="{y-6}" width="22" height="12" rx="2" fill="#EEEAE0"/>'
    s += text(hx + 40, hy + 14, "Powered", "ts") + text(hx + 40, hy + 27, "USB hub", "ts")
    # fire path, shifted right
    g, R, E = logic_side("", host=False)
    com, nc, no = R["K3_0"], R["K3_1"], R["K3_2"]
    g += text(no[0] + 14, no[1] + 4, "NO", "pl") + text(com[0] + 14, com[1] + 4, "COM", "pl") + text(nc[0] + 14, nc[1] + 4, "NC", "pl")
    jb, J = junction_board(1000, 560, flip=True)
    bp, BP = battery_compact(900, 400)
    ps, PS = power_switch(1040, 420)
    g += jb + bp + ps
    g += text(1115, 736, "flywheel + pusher motors", "ts", "middle")
    g += wire([BP["-"], (BP["-"][0], 540), (J["BAT-"][0], 540), J["BAT-"]], "black")
    g += wire([BP["+"], (BP["+"][0], 500), (PS["a"][0], 500), PS["a"]], "red")
    g += wire([PS["b"], (PS["b"][0], 520), (J["BAT+"][0], 520), J["BAT+"]], "red")
    g += wire([J["H_red"], (J["H_red"][0], no[1]), no], "red")
    g += wire([J["H_black"], (J["H_black"][0], nc[1]), nc], NC_COLOR)
    g += wire([J["H_blue"], (J["H_blue"][0], com[1]), com], C_COLOR)
    g += wire([J["H_white"], (J["H_white"][0], 546), (1272, 546), (1272, 760), (860, 760), (860, no[1])], "white", ends=False)
    g += f'<circle cx="860" cy="{no[1]}" r="5.5" fill="#6B7178"/>'
    g += text(866, 350, "white joins red", "ts") + text(866, 364, "(same NO screw)", "ts")
    g += text(900, 96, "Blaster harness → relay", "tb")
    g += text(900, 114, "red + white → NO (always rev)", "t")
    g += text(900, 131, f"{C_WIRE} (C, pusher motor) → COM", "t")
    g += text(900, 148, f"{NC_WIRE} (NC, brake) → NC", "t")
    g += note(900, 166, 330, ["Screw order varies by relay board: find COM", "with a meter before wiring. IN1–IN8 are bridged."])
    g += note(620, 662, 226, [f"{C_WIRE} = C → COM, {NC_WIRE} = NC → NC.", "Swapped = battery short."], warn=True)
    s += f'<g transform="translate({DX},0)">{g}</g>'
    # hub → ESP32 USB (ESP32 pins are in the shifted frame)
    ex, ey = E["USB"][0] + DX, E["USB"][1]
    s += wire([(hx + 30, 622), (ex, 622), (ex, ey)], "grey")
    s += text(hx + 40, 616, "USB serial · 115200", "ts")
    return s + "</svg>"


if __name__ == "__main__":
    jobs = [("wiring.svg", diagram_rig),
            ("wiring-blaster-stock.svg", diagram_stock)]
    for name, fn in jobs:
        (OUT / name).write_text(fn())
        print("wrote", OUT / name)
