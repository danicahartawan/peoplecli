"""
train the model, then emit the data block the canvas reads.

one source of truth: data/places.csv + data/annotations_guessed.csv feed both
the map and the scores, so a dot on screen and the number in its card are
always the same place. writes web/data.js, which index.html inlines.
"""

import csv, json, sys, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.scnam import PartialSCNAM
import sim.simulate as S

torch.manual_seed(0)

# the subset that goes on the map. fewer than the 45 we annotate, so the
# canvas stays legible.
SHOWN = ["p01","p03","p04","p06","p07","p08","p11","p13","p14","p15","p16",
         "p18","p21","p22","p23","p24","p25","p26","p27","p30","p31","p32",
         "p33","p34","p35","p37","p38","p39"]

CAT = {
 "p01":"quiet","p03":"enclosed","p04":"enclosed","p06":"hum","p07":"quiet",
 "p08":"quiet","p11":"hum","p13":"light","p14":"enclosed","p15":"enclosed",
 "p16":"light","p18":"light","p21":"enclosed","p22":"open","p23":"greenery",
 "p24":"greenery","p25":"greenery","p26":"greenery","p27":"greenery",
 "p30":"open","p31":"open","p32":"open","p33":"open","p34":"hum","p35":"hum",
 "p37":"hum","p38":"enclosed","p39":"light",
}
BLURB = {
 "p01":"long tables, high ceilings, everyone reading in the same direction.",
 "p03":"wood panels, lamps, armchairs. no laptops, so no clatter.",
 "p04":"underground. no daylight at all, and nobody talks.",
 "p06":"steady low buzz. easy to disappear into if you like company nearby.",
 "p07":"wide light, almost no foot traffic past the second floor.",
 "p08":"bag check at the door. everything here moves at half speed.",
 "p11":"talk, espresso, chairs scraping. good when silence feels worse.",
 "p13":"tall volume under the skeleton. warm and busy in the middle.",
 "p14":"concrete pocket with benches. loud outside, muffled inside.",
 "p15":"busy between the hour, then completely still.",
 "p16":"glass corner, long views west, mostly silent after dark.",
 "p18":"domed skylights and stone. almost nobody sits here.",
 "p21":"small shaded square, fountain, almost always free seats.",
 "p22":"flat open grass with the whole sky over it.",
 "p23":"sunken lawn under old trees, tucked off the main paths.",
 "p24":"tallest hardwoods around. cool, dim, smells like the coast.",
 "p25":"water noise covers everything else. benches along the bank.",
 "p26":"a hill nobody crosses on purpose. views west, mostly empty.",
 "p27":"small lawn behind the oldest building on campus.",
 "p30":"wide stone, long sight lines, bells on the hour.",
 "p31":"a big quiet oval. rarely more than a few people.",
 "p32":"empty stone tiers most days. sit high up and the city opens.",
 "p33":"turf and fences. quiet outside practice hours.",
 "p34":"the loudest place on campus. tables, flyers, drums.",
 "p35":"open terrace, food, constant crossing traffic.",
 "p37":"terraced steps, coffee, low conversation all day.",
 "p38":"east edge of campus, shaded, up against the hill.",
 "p39":"uphill, warm brick, quiet in the middle of the day.",
}
TAGS = {
 "p01":["reading","focus"],"p03":["reading","decompress"],"p04":["focus"],
 "p06":["focus","company"],"p07":["focus","reading"],"p08":["reading"],
 "p11":["company","brainstorm"],"p13":["light"],"p14":["focus"],
 "p15":["focus"],"p16":["focus","light"],"p18":["think","light"],
 "p21":["calm","reading"],"p22":["decompress","social"],
 "p23":["calm","decompress"],"p24":["calm","walk"],"p25":["calm","walk"],
 "p26":["solitude"],"p27":["calm"],"p30":["walk","think"],
 "p31":["walk","think"],"p32":["solitude","decompress"],"p33":["decompress"],
 "p34":["social"],"p35":["social","company"],"p37":["company","brainstorm"],
 "p38":["solitude"],"p39":["calm","reading"],
}

SLOT_LABEL = {
 "A_wkdy_morning":"weekday morning","B_wkdy_midday":"weekday midday",
 "C_wkdy_afternoon":"weekday afternoon","D_wkdy_evening":"weekday evening",
 "E_wkend_midday":"weekend midday",
}

# how a feature reads in a sentence. keyed on the OBSERVED value, not on the
# sign of the contribution -- the phrase describes the place, the sign only
# decides whether it counts for or against. keying it on the sign produced
# "talking is normal there" as a reason to go somewhere and read, because a
# silent room scores positively for reading.
PHRASE = {
 ("noise","hi"):"it is loud",
 ("noise","lo"):"it is quiet",
 ("noise_var","hi"):"the noise comes in bursts",
 ("noise_var","lo"):"the sound is steady rather than sudden",
 ("daylight","hi"):"there is daylight where you would sit",
 ("daylight","lo"):"there is no real daylight",
 ("greenery","hi"):"there is planting in view",
 ("greenery","lo"):"there is nothing green in view",
 ("enclosure","hi"):"it is enclosed",
 ("enclosure","lo"):"it is open on all sides",
 ("privacy","hi"):"nobody passes behind you",
 ("privacy","lo"):"you are exposed to foot traffic",
 ("crowding","hi"):"it is busy at that hour",
 ("crowding","lo"):"it is nearly empty at that hour",
 ("seat_comfort","hi"):"the seating is comfortable for a long sit",
 ("seat_comfort","lo"):"the seating is hard or absent",
 ("talk_ok","hi"):"talking is normal there",
 ("talk_ok","lo"):"it is a silent room",
 ("complexity","hi"):"there is a lot competing for attention",
 ("complexity","lo"):"there is little to look at",
 ("thermal","hi"):"it is comfortable to sit in",
 ("thermal","lo"):"it is cold or exposed",
}

QUERIES = [
 dict(id="read",  prompt="i am overstimulated and need to read for an hour",
      ctx=dict(arousal=.88, valence=.30, tol=.15, task="read", ns=.8, ex=.4)),
 dict(id="focus", prompt="i feel flat and i have to start an essay",
      ctx=dict(arousal=.15, valence=.35, tol=.70, task="focus", ns=.3, ex=.5)),
 dict(id="talk",  prompt="i want to think out loud with a friend",
      ctx=dict(arousal=.50, valence=.75, tol=.60, task="brainstorm", ns=.4, ex=.8)),
 dict(id="calm",  prompt="i need to calm down before a meeting",
      ctx=dict(arousal=.85, valence=.30, tol=.20, task="calm_down", ns=.6, ex=.4)),
]


def main():
    places_meta = {r["place_id"]: r for r in csv.DictReader(open(ROOT/"data/places.csv"))}
    feats, meta = S.load()
    ids = [r["place_id"] for r in csv.DictReader(open(ROOT/"data/annotations_guessed.csv"))]
    slots = [r["slot"] for r in csv.DictReader(open(ROOT/"data/annotations_guessed.csv"))]

    # train the model we settled on
    gen = torch.Generator().manual_seed(7)
    users = S.make_users(80, gen)
    tr_all = S.sessions(users[:64], 30, feats, gen)
    n = int(tr_all["y"].shape[0] * .85)
    tr = {k: v[:n] for k, v in tr_all.items()}
    va = {k: v[n:] for k, v in tr_all.items()}
    FREE = [S.FEATURES.index("noise"), S.FEATURES.index("daylight")]
    model = S.fit(PartialSCNAM(S.D, S.C, FREE), tr, va, use_uid=False)
    acc = S.accuracy(model, va, hard=True)
    print(f"trained partial model — close-call accuracy {acc*100:.1f}%")

    rows_for = lambda pid: [i for i, p in enumerate(ids) if p == pid]

    out_places = []
    for pid in SHOWN:
        m = places_meta[pid]
        out_places.append(dict(
            id=pid, name=m["name"], lat=float(m["lat"]), lng=float(m["lng"]),
            cat=CAT[pid], blurb=BLURB[pid], tags=TAGS[pid],
        ))

    out_queries = []
    with torch.no_grad():
        for q in QUERIES:
            ctx = S.query(**q["ctx"])
            # centre against the average place-slot under THIS state, so a
            # contribution reads as "vs a typical berkeley space, for you, now"
            base = model.contributions(ctx.repeat(feats.shape[0], 1), feats).mean(0)
            all_scores = model(ctx.repeat(feats.shape[0], 1), feats) - base.sum()

            per_place = {}
            for pid in SHOWN:
                idx = rows_for(pid)
                best = max(idx, key=lambda i: all_scores[i].item())
                phi = model.contributions(ctx, feats[best:best+1])[0] - base
                order = phi.abs().argsort(descending=True)[:4]
                per_place[pid] = dict(
                    fit=round(all_scores[best].item(), 3),
                    slot=SLOT_LABEL[slots[best]],
                    why=[dict(f=S.FEATURES[j], phi=round(phi[j].item(), 3),
                              obs=round(feats[best, j].item(), 2),
                              text=PHRASE[(S.FEATURES[j],
                                           "hi" if feats[best, j] > .5 else "lo")])
                         for j in order],
                )

            ranked = sorted(SHOWN, key=lambda p: -per_place[p]["fit"])
            top = ranked[0]
            pos = [w for w in per_place[top]["why"] if w["phi"] > 0][:2]
            reply = (f"<b>{places_meta[top]['name']}</b>, "
                     f"{per_place[top]['slot']}. " + ", and ".join(w["text"] for w in pos) + ".")
            out_queries.append(dict(id=q["id"], prompt=q["prompt"], reply=reply,
                                    top=top, ranked=ranked, places=per_place))
            print(f"  {q['id']:<6} -> {places_meta[top]['name']}")

    js = ("/* generated by sim/export_web.py — do not edit by hand */\n"
          f"const MODEL_ACC = {acc:.3f};\n"
          f"const PLACES = {json.dumps(out_places, indent=1)};\n"
          f"const QUERIES = {json.dumps(out_queries, indent=1)};\n")
    (ROOT/"web/data.js").write_text(js)
    print(f"\nwrote web/data.js ({len(js)//1024} kb)")


if __name__ == "__main__":
    main()
