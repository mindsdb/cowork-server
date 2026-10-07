---
name: connected-image-generation
description: 'MANDATORY reading before generating an image (photo, picture, illustration, logo, artwork) from a text prompt, and before building or testing a skill that generates images. There is no built-in image model: images are generated only through the user''s own Replicate account. Covers offering the Replicate connection when it is missing and making the real image when it is connected. Never fake an image with hand-drawn SVG, matplotlib, or a public image API.'
metadata:
  display_name: Generate images (connected service)
  provenance: host
---
IMAGE GENERATION — there is no built-in image model. The only way to produce a real image is the user's own Replicate account.

1. CHECK THE CONNECTION. Look for an env var named `DS_REPLICATE_<NAME>__API_TOKEN` in the connected data sources listed in your context. If none is listed, Replicate is not connected: go to step 2. If one is listed, go to step 3.

2. NOT CONNECTED — OFFER, DO NOT IMPROVISE.
   - In your first reply, say plainly that image generation is not built in and that it works through Replicate with the user's own token.
   - Say it is a paid service, billed per image to their Replicate account (small models cost a fraction of a cent per image), and that a small amount of prepaid credit acts as the spending cap.
   - Ask whether to set it up. On yes: recall the `connect-datasource` skill, then call `lookup_connector(id="replicate")` and follow that skill to render the form.
   - Do NOT try public or keyless image APIs, do NOT draw the image with SVG, matplotlib, PIL shapes, or HTML/CSS, and do NOT loop through alternatives. A missing capability is not a failure to work around.

3. CONNECTED — GENERATE.
   - Call `create_artifact(type="image", name=<short name>)` and write into the folder it returns.
   - In ONE scratchpad cell (stdlib only, nothing to install):

     ```python
     import json, os, urllib.request
     token = os.environ[TOKEN_VAR]
     req = urllib.request.Request(
         "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
         data=json.dumps({"input": {"prompt": PROMPT, "aspect_ratio": "1:1",
                                    "output_format": "png", "num_outputs": 1}}).encode(),
         headers={"Authorization": f"Bearer {token}",
                  "Content-Type": "application/json", "Prefer": "wait=60"},
     )
     prediction = json.load(urllib.request.urlopen(req, timeout=90))
     print(prediction["status"], prediction.get("error"))
     if prediction["status"] == "succeeded":
         urllib.request.urlretrieve(prediction["output"][0], os.path.join(FOLDER, "image.png"))
     ```

     Replace `TOKEN_VAR` with the exact env var name listed for the Replicate connection, and `PROMPT` and `FOLDER`. Never print the token. `aspect_ratio` also accepts "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"; set `num_outputs` up to 4 when the user wants options.
   - If `status` is still `starting` or `processing`, poll `prediction["urls"]["get"]` with the same Authorization header every few seconds, for at most one more minute.
   - Look at the result with `read_image` when it is available, and regenerate once with a sharper prompt if it clearly misses the request.

4. ERRORS. Stop and tell the user; do not switch to another provider or a workaround.
   - 401: the token is invalid or revoked. Offer to reconnect Replicate.
   - 402 or a billing/credit message: the Replicate credit is used up. Tell them to add credit under Account → Billing.
   - 422: the input was rejected; fix the input once, then report.
   - A `failed` status with an NSFW or safety error: say the prompt was refused and suggest a rewording.

BUILDING A SKILL THAT MAKES IMAGES. When the user asks for a skill whose output is an image (brand-style product shots, avatars, icons), the new skill states its own style rules and recalls `connected-image-generation` for the generation step instead of copying API code. Test it through this skill; if Replicate is not connected, follow step 2 instead of trying other image APIs.
