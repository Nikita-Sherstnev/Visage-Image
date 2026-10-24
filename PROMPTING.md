# Prompting

The DiT was trained on three prompt styles, mixed 50/50 between long and short from step 0:

- **Long:** a comma-separated list of attribute clauses in a fixed order. These follow best and look most photographic.
- **Short:** one sentence, e.g. `A bald man with a full beard, mouth open, sweaty face against a dark background with blue lights`.
- **Casual:** loose phrasing, e.g. `woman sticking her tongue out, outdoors in a park`.

Bare prompts like `a woman` also work. The model fills in the rest from the training distribution.

## Long caption format

Every clause is optional. Leave a slot out to let the model choose it, or to get its default (sharp, facing the camera, looking into the lens). The caption starts with a capital letter and ends with a period.

The clauses always appeared in this order during training:

| # | Slot | Written as | Common values |
|---|------|------------|---------------|
| 1 | sharpness | bare value | *(omit for sharp)*, `Slightly soft`, `Blurry` |
| 2 | subject | bare value | `a woman`, `a man in his 40s`, `a white woman in her 30s`, `a Latina woman`, `an Asian woman in her 20s`, `a Black woman`, `a mixed-race woman`, `a Middle Eastern man in his 30s`. Ages run from 20s to 70s, by decade |
| 3 | head turn | bare value | *(omit for facing)*, `turned three-quarter`, `in profile` |
| 4 | head pitch | bare value | `chin up`, `slightly chin down`, `chin down` |
| 5 | head tilt | bare value | `slight tilt to the left`, `tilted to the right` |
| 6 | gaze | bare value | *(omit for eye contact)*, `looking away`, `looking down`, `looking up`, `eyes closed` |
| 7 | framing | bare value | *(omit for a head-and-shoulders crop)*, `tight close-up`, `extreme close-up` |
| 8 | photo type | bare value | `studio photo`, `amateur phone photo`, `video frame`, `black-and-white photo` |
| 9 | action | bare value | `smiling broadly`, `taking a mirror selfie`, `hand raised to head`, `singing into a microphone`, `screaming with mouth wide open`, `extending tongue out of open mouth`, `resting chin on hand` |
| 10 | skin | bare value | `fair`, `fair, smooth`, `light`, `light brown`, `light tan`, `tan`, `olive`, `medium brown`, `dark`, `fair with visible freckles`, `tan with visible texture`, `sweaty, glistening` |
| 11 | hair | `hair …` | `hair long, wavy, blonde`, `hair dark, pulled back`, `hair short, black, curly`, `hair bald`; or as a phrase: `long, straight black hair, center part` |
| 12 | eyes | `eyes …` | `eyes brown`, `eyes blue`, `eyes dark brown`, `eyes green` |
| 13 | eyebrows | `eyebrows …` | `eyebrows dark, defined`, `eyebrows natural, medium thickness`, `eyebrows dark, thick, furrowed` |
| 14 | facial hair | bare value | `light stubble`, `short dark beard and moustache`, `full dark beard`, `short grey stubble` |
| 15 | nose | `nose …` | `nose small`, `nose narrow`, `nose ring`, `septum piercing` |
| 16 | lips | `lips …` | `lips closed, natural`, `lips open, revealing upper teeth`, `lips slightly parted`, `lips closed, slight smile`, `lips open in a wide smile, upper teeth visible`, `lips red lipstick` |
| 17 | expression | bare value | `happy`, `playful`, `joyful`, `content`, `calm`, `focused`, `intense`, `surprised`, `serious`, `contemplative`, `serene` |
| 18 | makeup | `makeup …` | `makeup subtle, natural`, `makeup dark eyeliner, mascara`, `makeup winged eyeliner`, `red lipstick` |
| 19 | clothing | bare value | `bare shoulders`, `white top`, `black top`, `white collared shirt`, `dark suit jacket, white collared shirt`, `denim jacket`, `thin gold chain necklace` |
| 20 | other people or body parts | `also in frame …` | `also in frame her hand holding the phone`, `also in frame blurred figures in background` |
| 21 | objects | `objects …` | `objects small gold hoop earrings`, `objects black-rimmed glasses`, `objects microphone` |
| 22 | background | `background …` | `background plain light-colored wall`, `background out of focus green foliage`, `background solid black`, `background dark wood paneling`, `background beach, sand, ocean, sky` |
| 23 | lighting | `lit by … light` | `lit by soft, even light`, `lit by bright, natural light`, `lit by soft, warm light`, `lit by natural daylight`, `lit by soft, directional light` |

**Clause order.** The training captions always used the order above, but moving clauses mostly still works: pose, gaze, expression, glasses and framing followed the prompt in every order tested (8 prompts × 8 seeds, original vs shuffled vs reversed). Two problems showed up:

- **Bare skin values get misread when moved.** `tan with visible texture` or `light brown` away from its slot was drawn as a tan fur or shearling collar (6/8 seeds shuffled and 8/8 reversed on one prompt). Keep skin right after the photo type, or write it as `tan skin`.
- **Opening with lighting and background** (fully reversed) weakened global style: `black-and-white photo` came out in colour on most seeds, and backgrounds got busier.

Keep the sharpness, subject and photo type near the start and the rest can go in any order.

The value lists are the most frequent training values. Close variants and new combinations work, because the encoder is a language model, but frequent wording follows most reliably. Within a slot, describe attributes in plain adjectives the way a photo caption would.

## Examples

```text
A woman in her 20s, amateur phone photo, fair with visible freckles, hair red, long, wavy, eyes green, lips open in a wide smile, upper teeth visible, joyful, makeup subtle, natural, white top, background out of focus green trees, lit by bright, natural light.

A man in his 30s, slightly chin down, studio photo, tan with visible texture, hair short, black, eyes brown, eyebrows dark, thick, furrowed, short dark stubble, lips closed, pressed together, angry, denim jacket, background out of focus red brick wall, lit by natural daylight.

A man in his 60s, slightly chin down, slight tilt to the left, studio photo, fair, hair light brown, short, receding, eyes blue, eyebrows natural, lips open, revealing upper teeth, happy, dark suit, white shirt, gold tie, background dark wood paneling with vertical grooves, lit by soft, even light.

A woman, in profile, chin down, tilted to the left, eyes closed, amateur phone photo, light, hair brown, pulled back into a messy bun, eyebrows dark, defined, lips closed, relaxed, black top, thin necklace, background bright blue sky with wispy white clouds, lit by bright, natural, outdoor light.

An Asian woman, turned three-quarter, slightly chin down, tilted to the left, studio photo, light brown, hair dark, wavy, shoulder-length, eyes brown, eyebrows dark, medium thickness, nose ring, lips closed, glossy pink lipstick, objects small gold stud earring, background solid muted green-grey, lit by soft, even light.
```

The first two are showcase tiles (`scripts/make_showcase.py`, seeds 41 and 7).

## Tips

- **Sharpness:** leave the first slot empty for sharp images. `Slightly soft` and `Blurry` reproduce soft source photos.
- **Photo type** sets the overall look more than any other single slot. `studio photo` is clean and evenly lit, `amateur phone photo` is a casual selfie look, and `video frame` is a webcam or stream look.
- **Expressions** follow best when the lips slot agrees with them: `lips open, upper teeth visible, laughing` beats `laughing` on its own.
- **Tongues:** write `tongue out` or `sticking tongue out`. `tongue visible` mostly fails.
- **Weak spots:** pose and gaze follow less reliably than appearance does. Sunglasses, wet hair and lip biting mostly fail. "Biting lip" never appears in a training caption.
- **Out of domain:** the model was trained on photos of faces. Paintings, fantasy styles and full-body shots are outside the training data.
