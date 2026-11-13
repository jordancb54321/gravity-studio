# Gravity Studio

A native Python 2D gravity sandbox.
NumPy handles particle physics; Qt provides native controls and drawing.
The application works locally without network requests.

## Run

    ./run.sh

Or run `.venv/bin/python -m gravity_studio` from this directory.

To reproduce the environment with Python 3.12+ and uv:

    uv sync --locked --extra dev
    uv run gravity-studio

The runtime dependencies are NumPy and PySide6-Essentials (Qt Widgets).
The development dependencies are only needed for tests and quality checks.

## Use

- Click “Hide settings” beside the title to collapse the whole side panel and
  give the canvas more room. “Show settings” brings it back.
- Choose a task at the top of the panel: Build a scene (the starting view),
  Paint with gravity, Adjust physics, or Manage scene. A short hint explains
  each task. All settings shows the complete panel. Switching tasks keeps your
  values, folded categories and running scene unchanged.
- The wider settings panel keeps labels and number fields aligned, with paired
  tool and clear buttons to reduce scrolling. Scroll to reach all expanded options.
- Click any category heading to expand or collapse it: Place, Physics, Painting,
  Image painting, Random balls and Scene. Categories open independently and keep
  their values when folded. Tab to a heading and press Space to toggle it.
- Select the gravity-dot tool, then click the canvas to place fixed attractors.
- Select the ball tool and click to place balls. New balls start at rest.
- Select or drag an existing gravity dot to edit or move it.
- Choose a colour for the selected gravity dot and for new gravity dots.
- Adjust air resistance while the simulation is running: zero means no drag;
  higher values damp velocity more quickly. Gravity strength is adjustable too.
- Enable “Paint spawn points” to leave a permanent speck at each captured ball's
  ORIGINAL spawn position, in the colour of the gravity dot that caught it.
  The ball disappears on capture whether or not painting is enabled.
- Under Painting, adjust “Origin mark size” from 1 to 100 px (default 2.5 px).
  Use smaller marks for fine detail or larger marks for broader coverage.
  This redraws existing marks and
  applies to future captures without changing balls, gravity dots or physics.
  Reset restores 2.5 px.
- Under Random balls, enable “Continuous stream” and set “Balls per second”
  (1–10,000; default 100) for a constant random feed. The rate can change live.
  Streaming pauses with the simulation. Turning the toggle off stops new stream
  balls immediately; Clear balls and Reset also stop the stream. Batch spawning
  remains available independently, including while paused.
- Spawn random batches repeatedly. There is no fixed total dot or ball cap.
  Large batches are queued in chunks, can be cancelled, and still depend on RAM
  and processing power. More balls AND more dots require more computation.
  The continuous stream targets wall-clock seconds while unpaused, keeps fractional
  balls across frames, and does not accumulate a backlog while paused or disabled.
  After stalls it considers at most 0.25 seconds and adds at most 1,000 balls per
  frame (shared with manual batches), skipping excess stream spawns to stay
  responsive. Actual rates therefore depend on available processing capacity.
- Pause to arrange a scene; resume to watch it evolve.
- Clear balls and clear paint independently. Reset clears the whole scene.

Paint uses the gravity dot's colour at the instant of capture. Later colour
changes do not recolour existing specks; switching painting off keeps existing
specks and only stops future painting. Removing a dot does not erase its paint.

All scene data is in memory. Closing the app discards the scene and paint.
There are no persistence or export guarantees.

## Image painting (colour-directed gravity)

1. Under Image painting, choose the palette size with “Colours on load” (minimum
   2; default 8). The old 32-colour cap is removed. The native number field accepts
   up to 2,147,483,647; the actual palette cannot exceed the image's distinct RGB
   colours and may contain fewer colours than requested.
2. Click Load image and choose a PNG, JPEG, BMP or WebP. A nonempty scene is
   replaced only after confirmation. Loading extracts a palette, places a
   gravity dot at the centroid of each colour class, enables Image painting
   mode and enables Paint spawn points.
3. Under Random balls, enable Continuous stream or spawn batches. Every new
   ball samples its image colour class at its ORIGINAL spawn location and is
   pulled/captured only by dots in that class. Its mark still comes from the
   actual capturing dot; the app does not draw the source image as a reveal.
4. Untick Show gravity dots to hide every marker and selection ring. Gravity,
   captures, and paint continue unchanged. Hidden dots are not mouse-editable;
   show them again to select, drag or erase them.

In image mode, Continuous stream gently favours high-contrast edges and colour
changes detected in the imported working image before palette reduction. It
chooses between two random locations with a small contrast preference, so flat
areas still receive particles and subtle noise is not exaggerated. This is
automatic: there is no extra toggle. Manual batches and sandbox mode remain
uniformly random, and stream rates, gravity and capture behaviour are unchanged.

The result is a palette-reduced, point-painted approximation. Basic flat-colour
images work especially well. An original example is included at
`assets/heart-demo.png`. There is no cloud upload or image-generation service.

Images fit the canvas with aspect ratio preserved. In image mode, random balls
spawn only inside the fitted image. Manually placed balls outside that rectangle
use normal sandbox gravity. The fit is fixed at import time: resizing does not
stretch existing paint or reinterpret old spawn positions. Reload to fit a new
canvas size. Images retain their original pixel resolution: there is no automatic
256-pixel reduction or application pixel-count ceiling. The 32 MiB file-size
check and Qt's decoder memory protections still apply. Transparent pixels are
composited onto black; static/first-frame image data is used.

Large images and palettes require more memory and loading time. Each populated
palette class creates a gravity dot, so increasing colours also increases physics
work. Loading is synchronous and can temporarily make the window unresponsive;
available memory and processing power remain practical limits.

Moving an image dot keeps its palette class. Recolouring one changes that
class's colour across its image dots and future captures; existing marks do not
change. Deleting the last dot of a class leaves its balls uncaptured until a
matching-colour source is added again. Turning Image painting mode off restores
ordinary all-dot gravity without deleting the loaded field or current paint.
Reset clears the imported image too. Palette size changes apply on the next
image load, not retroactively.

## Physics

Dots are fixed point attractors; their pull combines. Balls do not attract or
collide with each other. Gravity uses a softened inverse-square force to avoid
singularities near attractor centres (softening is at least half a pixel).
Balls are captured on touching a dot.
Linear air resistance damps velocity exponentially, rather than depending on
the display frame rate. For numerical safety, the model rejects gravity or drag
outside 0..1e9 and limits velocity components to 1,000,000 pixels/second; these
limits do not cap the number of objects. The GUI uses a fixed 1/120-second physics timestep and
limits catch-up work so an overloaded scene slows simulation time rather than
spiralling into unlimited catch-up steps. The canvas boundary reflects balls.
This is an interactive approximation, not a scientific integrator.

A swept segment/circle test catches fast crossings of gravity dots and resolves
the earliest hit rather than choosing whichever dot was inserted first. Paths
that bounce off a wall are split into their reflected segments before capture
selection, including corner bounces.
Positions, velocities and immutable spawn positions are stored in growing NumPy
buffers; no arbitrary total entity cap is built into the model. Resizing clamps
current objects into the new bounds but does not move historical spawn points
or paint. Paint outside a smaller canvas becomes visible again if it is enlarged.

## Verify

    ./check.sh

Runs Ruff formatting and lint checks, mypy, and model/Qt tests. Qt tests use an
isolated offscreen platform, not the user's live desktop. This does not itself
prove that a real Wayland window opens: that is a separate smoke check.

Tests cover gravity direction/superposition, drag, capture colour and spawn
coordinates, toggling, fast crossings, overlapping sources, random batches,
resizing, reset/removal, input validation and actual Qt control interactions.

Implementation layout:

- gravity_studio/model.py — physics and scene state, independent of Qt.
- gravity_studio/image_field.py — palette extraction and spawn colour-class mapping.
- gravity_studio/image_io.py — bounded local image decoding and alpha compositing.
- gravity_studio/ui.py — native controls, canvas, rendering and fixed-step loop.
- gravity_studio/__main__.py — module entry point.
- tests/ — deterministic model and offscreen GUI checks.
- uv.lock — dependency resolution for reproducible installation.
