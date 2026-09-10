"""
fill the annotation sheet with GUESSED values.

this is stand-in data so the pipeline can be built and the ui can be wired up
before anyone walks around with a light meter. every number here is my estimate
from knowing the buildings, not a measurement. the point is that the columns,
ranges and correlations are realistic enough that code written against this file
keeps working when real rows replace it.

values are built from an archetype per place category, a per-place override for
the ones with a distinctive character, and a slot modifier for time of day.
that structure matters: it means noise and crowding move together with the hour
the way they really do, instead of being independent noise.
"""

import csv, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLOTS = ["A_wkdy_morning", "B_wkdy_midday", "C_wkdy_afternoon",
         "D_wkdy_evening", "E_wkend_midday"]

# archetype: db, dbvar, lux, kelvin, seats, enclos, cplx, green, daylt,
#            seat, priv, talk, therm, outlets, wifi, food, wc, table, backwall
ARCH = {
 "study_formal": (42,2, 420,4000, 120,4,2,1,4, 3,3,1,4, 1,1,0,1,1,1),
 "study_quiet":  (38,1, 380,4200,  90,4,2,1,3, 3,4,1,4, 1,1,0,1,1,1),
 "study_soft":   (36,1, 180,2900,  40,5,3,1,2, 5,4,1,4, 0,1,0,1,1,1),
 "study_social": (58,4, 450,4500, 140,4,3,1,3, 3,2,4,4, 1,1,1,1,1,0),
 "cafe":         (62,4, 520,3600,  45,3,4,2,4, 4,2,5,4, 1,1,1,1,1,0),
 "library":      (44,2, 400,4100,  80,4,3,1,4, 3,3,2,4, 1,1,0,1,1,1),
 "lobby":        (52,4, 900,5200,  25,3,4,2,5, 2,2,3,3, 0,1,0,1,0,0),
 "lounge":       (50,3, 520,3900,  35,4,4,2,4, 5,3,4,4, 1,1,0,1,1,1),
 "corridor":     (46,5, 300,4300,  14,3,3,1,3, 2,2,3,3, 0,1,0,1,0,1),
 "courtyard":    (50,3,3800,5600,  20,3,3,3,5, 3,3,4,3, 0,1,0,0,1,1),
 "plaza":        (58,4,8000,5800,  24,1,4,2,5, 2,1,5,3, 0,1,0,0,1,0),
 "gateway":      (64,5,7000,5700,   6,1,5,2,5, 1,1,5,3, 0,1,0,0,0,0),
 "lawn":         (44,3,9000,5700,   0,1,2,5,5, 2,2,4,3, 0,1,0,0,0,0),  # cap set below
 "field":        (42,3,9500,5800,  30,1,1,4,5, 2,2,3,3, 0,1,0,0,0,0),
 "grove":        (40,2,1400,5000,   8,3,2,5,3, 2,4,3,2, 0,0,0,0,0,0),
 "creek":        (46,2,2200,5200,  10,2,3,5,4, 3,4,3,3, 0,0,0,0,0,0),
 "hill":         (38,2,7500,5700,   4,1,2,4,5, 2,5,3,3, 0,0,0,0,0,0),
 "path":         (40,2,2600,5100,   4,2,3,5,4, 2,4,3,3, 0,0,0,0,0,0),
 "amphitheatre": (36,2,8500,5800, 300,1,2,3,5, 3,5,4,3, 0,0,0,0,0,0),
 "steps":        (42,3,2000,5200,  12,2,2,4,4, 2,4,3,3, 0,0,0,0,0,1),
 "terrace":      (44,3,5000,5600,  16,2,3,3,5, 3,4,4,4, 0,1,0,0,1,1),
 "trail":        (34,2,7000,5700,   0,2,2,5,5, 1,5,3,3, 0,0,0,0,0,0),
 "garden":       (36,2,5500,5400,  20,2,3,5,5, 3,4,3,3, 0,0,0,0,0,0),
 "park":         (52,4,8000,5700,  18,1,4,4,5, 2,2,4,3, 0,0,0,1,1,0),
}

# per-place overrides + how busy the place runs relative to its archetype
OVER = {
 "p03": dict(talk=1, seat=5, priv=5, cplx=2, pop=.45),   # morrison: armchairs, no laptops
 "p04": dict(daylt=1, lux=210, kelvin=3800, priv=4, pop=.85),  # stacks: no daylight at all
 "p05": dict(db=64, dbvar=5, talk=5, priv=1, pop=1.15),  # moffitt 1: talking floor
 "p06": dict(db=40, dbvar=2, talk=1, priv=3, pop=1.05),
 "p08": dict(db=32, dbvar=1, priv=4, cplx=2, pop=.30),   # bancroft: bag check, very slow
 "p11": dict(db=66, dbvar=5, pop=1.25),                  # fsm cafe
 "p16": dict(lux=1100, daylt=5, priv=4, pop=.55),        # soda 5th: glass corner
 "p18": dict(lux=1500, daylt=5, cplx=4, pop=.35),        # mining lobby: skylights
 "p22": dict(pop=1.30),                                  # memorial glade
 "p24": dict(lux=900, daylt=2, therm=2, pop=.30),        # eucalyptus: dim and cold
 "p25": dict(db=52, dbvar=1, pop=.35),                   # creek: water masks everything
 "p26": dict(pop=.12), "p31": dict(pop=.25), "p32": dict(pop=.15),
 "p34": dict(db=70, dbvar=5, cplx=5, pop=1.60),          # sproul
 "p35": dict(db=68, dbvar=5, pop=1.45),
 "p36": dict(pop=1.50),                                  # sather gate pinch point
 "p38": dict(pop=.15), "p41": dict(pop=.10),
 "p43": dict(cplx=5, priv=1, pop=.90),                   # peoples park
 "p45": dict(pop=.60),
}

#              occ,  db,  lux_in, lux_out, therm
SLOTMOD = {
 "A_wkdy_morning":   (.45, -4, .70, .55, -1.0),
 "B_wkdy_midday":    (1.0, +3, 1.0, 1.0, +0.4),
 "C_wkdy_afternoon":  (.85,  0,  .90, .85, +0.6),
 "D_wkdy_evening":    (.30, -7,  .85, .03, -1.4),
 "E_wkend_midday":    (.28, -6, 1.0, 1.0, +0.4),
}

# comfortable occupancy for spaces that have no seats to count. a lawn holds
# people without holding chairs, so crowding needs its own denominator.
CAP = {"lawn":60, "field":40, "trail":8, "plaza":90, "gateway":40, "park":50,
       "hill":10, "path":8, "grove":16, "creek":14, "amphitheatre":300,
       "steps":16, "terrace":20, "courtyard":30, "garden":25, "lobby":30,
       "corridor":16, "cafe":50, "lounge":40}

clamp = lambda v, lo, hi: max(lo, min(hi, v))
r5 = lambda v: int(round(clamp(v, 1, 5)))


def main():
    places = list(csv.DictReader(open(ROOT / "data/places.csv")))
    rng = random.Random(11)
    out = []

    for p in places:
        a = ARCH[p["category"]]
        (db, dbvar, lux, kelvin, seats, enclos, cplx, green, daylt,
         seat, priv, talk, therm, outlets, wifi, food, wc, table, bw) = a
        o = OVER.get(p["place_id"], {})
        db, dbvar = o.get("db", db), o.get("dbvar", dbvar)
        lux, kelvin = o.get("lux", lux), o.get("kelvin", kelvin)
        cplx, green = o.get("cplx", cplx), o.get("green", green)
        daylt, seat = o.get("daylt", daylt), o.get("seat", seat)
        priv, talk = o.get("priv", priv), o.get("talk", talk)
        therm = o.get("therm", therm)
        pop = o.get("pop", .6)
        indoor = int(p["indoor"])

        for slot in SLOTS:
            occ_m, db_m, lux_in, lux_out, therm_m = SLOTMOD[slot]
            j = lambda s: (rng.random() - .5) * s

            # crowding drives noise, not the other way round
            load = clamp(pop * occ_m + j(.10), 0, 1.5)
            s_tot = seats
            cap = seats if seats else CAP.get(p["category"], 20)
            s_occ = int(round(min(s_tot, s_tot * load))) if s_tot else 0
            people = int(round(cap * load * (1.0 if not s_tot else 0.25))) + s_occ

            # a room has a noise floor and a crowd has a ceiling: 30 dB is an
            # empty quiet room, 78 dB is as loud as a plaza of people gets
            dbm = clamp(db + db_m + 7 * (load - .6) + j(3), 30, 78)
            dbp = clamp(dbm + 4 + 3.2 * dbvar + j(4), 34, 92)

            lx = lux * (lux_in if indoor else lux_out) * (1 + j(.18))
            # outdoors after dark there is no daylight to rate
            dl = 1 if (not indoor and slot == "D_wkdy_evening") else daylt
            if indoor and slot == "D_wkdy_evening":
                dl = max(1, daylt - 2)
            kv = kelvin if not (slot == "D_wkdy_evening") else min(kelvin, 3200)

            out.append({
                "place_id": p["place_id"], "slot": slot,
                "date": "GUESS", "time": "", "rater": "guess_v1",
                "weather": "clear" if slot != "A_wkdy_morning" else "fog",
                "temp_f": int(round(58 + therm_m * 4 + j(4))),
                "lux": int(round(lx)), "kelvin": int(round(kv)),
                "db_mean": round(dbm, 1), "db_peak": round(dbp, 1),
                "db_variability": r5(dbvar + (load - .6) * 2),
                "seats_total": s_tot, "seats_occupied": s_occ, "capacity": cap,
                "people_visible": people,
                "enclosure": r5(enclos), "complexity": r5(cplx + (load - .6) * 1.5),
                "greenery": r5(green), "daylight": r5(dl),
                "seat_comfort": r5(seat), "privacy": r5(priv - (load - .5) * 2.2),
                "talk_ok": r5(talk), "thermal": r5(therm + therm_m),
                "outlets": outlets, "wifi": wifi, "food_drink": food,
                "restroom": wc, "table_surface": table, "back_to_wall": bw,
                "photo_wide": "", "photo_pov": "", "photo_light": "",
                "notes": "guessed, not measured",
            })

    cols = list(out[0].keys())
    with open(ROOT / "data/annotations_guessed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(out)
    print(f"wrote {len(out)} guessed rows -> data/annotations_guessed.csv")

    # sanity: the extremes should look like the real buildings
    for pid, slot in [("p04","B_wkdy_midday"), ("p34","B_wkdy_midday"),
                      ("p03","D_wkdy_evening"), ("p22","D_wkdy_evening")]:
        r = next(x for x in out if x["place_id"]==pid and x["slot"]==slot)
        nm = next(p["name"] for p in places if p["place_id"]==pid)
        print(f"  {nm[:34]:<34} {slot[:12]:<12} "
              f"db {r['db_mean']:>5} daylight {r['daylight']} "
              f"crowd {r['seats_occupied']}/{r['seats_total']} priv {r['privacy']}")


if __name__ == "__main__":
    main()
