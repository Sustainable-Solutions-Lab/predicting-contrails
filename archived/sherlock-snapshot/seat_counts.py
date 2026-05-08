
icao_seat_capacity = {
    "SU95": 85,     # Sukhoi Superjet 100-95 (~75–95 → midpoint)
    "A319": 130,    # Airbus A319 (~110–140 seats)
    "A332": 275,    # Airbus A330-200 (~275 seats typical)
    "A343": 295,    # Airbus A340-300 (~295 seats)
    "A320": 155,    # Airbus A320 (~150–170 seats)
    "A321": 200,    # Airbus A321 (~185–230 seats)
    "A21N": 206,    # Airbus A321neo (~180–220 seats) :contentReference[oaicite:0]{index=0}
    "BCS3": 135,    # Airbus A220-300 (~120–150)
    "BCS1": 110,    # Airbus A220-100 (~100–120)
    "A359": 325,    # Airbus A350-900 (~300–350)
    "A333": 275,    # Airbus A330-300 (~275–300)
    "A346": 320,    # Airbus A340-600 (~320)
    "A388": 525,    # Airbus A380 (~500–550 typical)
    "MD11": 300,    # McDonnell Douglas MD‑11 (~300)
    "B738": 162,    # Boeing 737‑800 (~160–180)
    "B739": 175,    # Boeing 737‑900 (~170–190)
    "B734": 140,    # Boeing 737‑400 (~140)
    "B752": 165,    # Boeing 757‑200 (~165)
    "B753": 185,    # Boeing 757‑300 (~185)
    "B762": 210,    # Boeing 767‑200ER (~210)
    "B763": 243,    # Boeing 767‑300ER (~243) etc.
    "B772": 300,    # Boeing 777‑200ER (~300‑325)
    "B77W": 350,    # Boeing 777‑300ER (~350)
    "B789": 320,    # Boeing 787‑9 (~290‑320)
    "B78X": 242,    # Boeing 787‑10 (~290‑330) but approx.
    "B788": 250,    # Boeing 787‑8 (~240‑260)
    "CRJ9": 90,     # Bombardier CRJ‑900 (~85‑100)
    "CRJ7": 78,     # CRJ‑700 (~70‑80)
    "CRJ2": 50,     # CRJ‑200 (~50)
    "CRJ1": 50,     # CRJ‑100 (~50)
    "E170": 76,     # Embraer E‑170 (~70‑80)
    "E175": 80,     # Embraer E‑175 (~80)
    "E190": 98,     # Embraer E‑190 (~90‑100)
    "E195": 106,    # Embraer E‑195 (~100–110)
    "E2TS": 110,    # Embraer E‑195‑E2 (~110)
    "E145": 50,     # Embraer ERJ‑145 (~50)
    "F100": 100,    # Fokker F100 (~100)
    "RJ1H": 100,    # Avro RJ85/RJ100 (~85‑100)
    "A306": 286,    # Airbus A300‑600 (~280)
    "A310": 240,    # Airbus A310 (~240)
    "C17": 134,     # Boeing C‑17 Globemaster III (military troop transport; typical seating ~134)
    "A339": 350,    # Airbus A330‑900neo (~300‑350)
    "A340": 300,    # Airbus A340 (general)
    "IL96": 262,    # Ilyushin Il‑96 (~262 typical)
    "DC10": 270,    # McDonnell Douglas DC‑10 (~270)
    "B744": 416,    # Boeing 747‑400 (~400‑450)
    "B742": 412,    # Boeing 747‑200 (~400)
    "B748": 467,    # Boeing 747‑8 (~450‑480)
    "A350": 330,    # A350‑900 (~325)
    "B767": 255,    # Boeing 767 (generic midpoint)
    "B777": 330,    # Boeing 777 (generic midpoint)
    "B787": 290,    # Boeing 787 (generic midpoint)
}

def get_seat_capacity(icao):
    if icao in icao_seat_capacity:
        return icao_seat_capacity[icao]
    # otherwise likely a small private jet
    return 10

