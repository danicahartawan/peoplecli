# annotation codebook

one row per **place × timeslot**. 45 places × 5 slots = 225 rows.
a place visited once is a weak data point — the same room at 9am and 9pm is
effectively two different places.

## slots
| code | when |
|---|---|
| A_wkdy_morning | weekday 9–11 |
| B_wkdy_midday | weekday 12–2 |
| C_wkdy_afternoon | weekday 3–5 |
| D_wkdy_evening | weekday 7–9 |
| E_wkend_midday | weekend 12–3 |

## photos
three per visit, same three every time:

1. `wide` — from the entrance, shows the whole space and how enclosed it is
2. `pov` — seated, eye level, what you actually look at while working
3. `light` — toward the main light source (window, skylight, sky)

name them `{place_id}_{slot}_{wide|pov|light}.jpg` → `p03_A_wide.jpg`

**turn on camera location.** EXIF then carries gps + timestamp for free, so you
never hand-enter coordinates or times, and you can verify a row was really
collected on site.

## measured (phone, ~90 seconds)
| field | how | unit |
|---|---|---|
| lux | light meter app, held at table height facing up | lux |
| kelvin | same app, color temperature | K |
| db_mean | sound meter, 60 seconds, average | dB |
| db_peak | same run, max | dB |
| temp_f | weather app or thermometer | °F |
| weather | clear / cloudy / fog / rain / wind | text |

`db_peak - db_mean` matters more than either alone. intermittent noise disrupts
far more than steady noise at the same average level, and almost nobody records it.

## counted
| field | how |
|---|---|
| seats_total | seats that exist in view |
| seats_occupied | seats with a person in them |
| people_visible | total humans in view, seated or not |

crowding is derived later as `seats_occupied / seats_total`. record the two
numbers, never the ratio — you lose the size of the space otherwise.

## rated 1–5
rate what is true **right now**, not what the place is usually like.

**enclosure** — how held-in the space feels
1 = open sky, no walls · 3 = walls on two sides · 5 = small closed room

**complexity** — how much visual stuff competes for attention
1 = blank wall, lawn, plain surfaces · 3 = ordinary furnished room
5 = posters, clutter, movement, many colors

**greenery** — living plants in view
1 = none · 3 = a few plants or a tree through a window · 5 = surrounded by planting

**daylight** — natural light, independent of brightness
1 = no daylight at all (windowless) · 3 = daylight reaches you indirectly
5 = direct sky visible from where you sit

**seat_comfort**
1 = no seating, standing or ground · 3 = hard chair and table
5 = soft chair, sofa, or a good ledge

**privacy** — can others see your screen and hear you
1 = fully exposed, traffic behind you · 3 = side exposure only
5 = back to wall, nobody passes

**talk_ok** — social permission to speak at normal volume
1 = silence enforced · 3 = quiet talking tolerated · 5 = talking is the norm

**thermal** — comfort right now
1 = too cold or too hot to stay · 3 = fine with a jacket · 5 = ideal

**db_variability**
1 = flat steady hum · 3 = occasional spikes · 5 = constant unpredictable bursts

## binary (0/1)
`outlets` `wifi` `food_drink` `restroom` `table_surface` `back_to_wall`
— all mean "available where you would actually sit," not somewhere in the building.

## rating drift
re-rate 10 rows a week later without looking at the originals. if your ratings
move more than 1 point on average, the subjective fields need tighter anchors
before you collect the rest. do this early, not at the end.

## why these fields and no others
every field here is a feature the model scores against **and** a phrase the app
can say back to the user. if a field cannot finish the sentence "this place fits
because ___," it does not belong in the schema.
