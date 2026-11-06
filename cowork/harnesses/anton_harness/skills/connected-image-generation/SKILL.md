---
name: connected-image-generation
description: 'MANDATORY reading before generating a photo, picture or illustration from a text prompt, and before building or testing a skill that does. There is no built-in image model: such images come only from the user''s own Replicate account. Covers offering the Replicate connection when it is missing and making the real image when it is connected. Vector graphics, icons, diagrams and charts drawn with code are not image generation and do not need this skill.'
metadata:
  display_name: Generate images (connected service)
  provenance: host
---
IMAGE GENERATION — there is no built-in image model. The only way to produce a real photo, picture or illustration is the user's own Replicate account.

Out of scope: when the user asks for an SVG, a vector logo or icon, a diagram, or a chart, draw it with code as usual. This skill is for raster images a generative model makes.

1. CHECK THE CONNECTION. Look for an env var named `DS_REPLICATE_<NAME>__API_TOKEN` in the connected data sources listed in your context. If none is listed, Replicate is not connected: go to step 2. If one is listed, go to step 3.

2. NOT CONNECTED — OFFER, DO NOT IMPROVISE.
   - In your first reply, say plainly that image generation is not built in and that it works through Replicate with the user's own token.
   - Say it is a paid service, billed per image to their Replicate account (small models cost a fraction of a cent per image), and suggest bounding the spend under Account → Billing on Replicate before connecting.
   - Ask whether to set it up. On yes: recall the `connect-datasource` skill, then call `lookup_connector(id="replicate")` and follow that skill to render the form.
   - In a chat channel (Slack, Telegram, etc.) a form cannot render: say the connection is set up in the Cowork desktop app instead.
   - Do NOT try public or keyless image APIs, do NOT fake the picture with SVG, matplotlib, PIL shapes, or HTML/CSS, and do NOT loop through alternatives. A missing capability is not a failure to work around.

3. CONNECTED — GENERATE.
   - Call `create_artifact(name=<short name>, description=<one line>, type="image", primary="image_0.png")` and write into the folder it returns.
   - In ONE scratchpad cell with `packages: ["requests"]`:

     ```python
     import os, requests
     headers = {"Authorization": f"Bearer {os.environ[TOKEN_VAR]}", "Prefer": "wait=60"}
     r = requests.post(
         "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
         json={"input": {"prompt": PROMPT, "aspect_ratio": "1:1",
                         "output_format": "png", "num_outputs": 1}},
         headers=headers, timeout=90,
     )
     if r.status_code >= 400:
         print("HTTP", r.status_code, r.text[:500])
     else:
         prediction = r.json()
         print(prediction["status"], prediction.get("error"))
         for i, url in enumerate(prediction.get("output") or []):
             with open(os.path.join(FOLDER, f"image_{i}.png"), "wb") as f:
                 f.write(requests.get(url, timeout=60).content)
     ```

     Replace `TOKEN_VAR` with the exact env var name listed for the Replicate connection, and `PROMPT` and `FOLDER`. Never print the token or the headers. `aspect_ratio` also accepts "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"; set `num_outputs` up to 4 when the user wants options (each one is billed).
   - If `status` is still `starting` or `processing`, poll `prediction["urls"]["get"]` with the same Authorization header every few seconds, for at most one more minute, then save the outputs the same way.
   - Look at the result with `read_image` when it is available, and regenerate once with a sharper prompt if it clearly misses the request.

4. ERRORS. Stop and tell the user; do not switch to another provider or a workaround.
   - 401: the token is invalid or revoked. Offer to reconnect Replicate.
   - 402 or a billing/credit message: the Replicate account cannot be billed. Tell them to check Account → Billing.
   - 422: the input was rejected; fix the input once using the printed detail, then report.
   - A `failed` status with an NSFW or safety error: say the prompt was refused and suggest a rewording.

BUILDING A SKILL THAT MAKES IMAGES. When the user asks for a skill whose output is a generated image (brand-style product shots, avatars, illustrations), the new skill states its own style rules and recalls `connected-image-generation` for the generation step instead of copying API code. Test it through this skill; if Replicate is not connected, follow step 2 instead of trying other image APIs.
