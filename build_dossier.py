from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path(r"D:\midori\Midori_Multimodal_Dataset_and_Model_Dossier.docx")
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
sec.top_margin, sec.bottom_margin = Inches(.62), Inches(.62)
sec.left_margin, sec.right_margin = Inches(.68), Inches(.68)
sec.header_distance, sec.footer_distance = Inches(.3), Inches(.3)

NAVY = "16324F"
TEAL = "167D8D"
PALE = "EAF2F5"
GRAY = "536273"

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Aptos"
normal.font.size = Pt(9.2)
normal.font.color.rgb = RGBColor.from_string("263645")
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.05
for name, size, color in [("Title", 25, NAVY), ("Heading 1", 17, NAVY), ("Heading 2", 11.2, TEAL), ("Heading 3", 9.5, NAVY)]:
    s = styles[name]
    s.font.name = "Aptos Display" if name != "Heading 3" else "Aptos"
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = RGBColor.from_string(color)
    s.paragraph_format.space_before = Pt(7 if name != "Heading 1" else 2)
    s.paragraph_format.space_after = Pt(3)

header = sec.header.paragraphs[0]
header.text = "MIDORI  /  MULTIMODAL ROBOT LEARNING"
header.style = styles["Normal"]
header.runs[0].font.size = Pt(8)
header.runs[0].font.bold = True
header.runs[0].font.color.rgb = RGBColor.from_string(TEAL)
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
r = footer.add_run("Design dossier  •  26 October 2026  |  ")
r.font.size = Pt(8); r.font.color.rgb = RGBColor.from_string(GRAY)
fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
footer._p.append(fld)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), fill); tcPr.append(shd)

def set_cell(cell, text, bold=False, color=None, size=8.1):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(1)
    run = p.add_run(str(text)); run.bold = bold; run.font.name = "Aptos"; run.font.size = Pt(size)
    if color: run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        set_cell(t.rows[0].cells[i], h, True, "FFFFFF", 8)
        shade(t.rows[0].cells[i], NAVY)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            set_cell(cells[i], val, False, None, 7.7)
            if ri % 2 == 1: shade(cells[i], "F3F7F9")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths): row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t

def p(text="", style=None, boldlead=None):
    para = doc.add_paragraph(style=style)
    if boldlead and text.startswith(boldlead):
        para.add_run(boldlead).bold = True
        para.add_run(text[len(boldlead):])
    else:
        para.add_run(text)
    return para

def bullets(items):
    for item in items:
        para = doc.add_paragraph(style="List Bullet")
        para.paragraph_format.left_indent = Inches(.2)
        para.paragraph_format.first_line_indent = Inches(-.12)
        para.paragraph_format.space_after = Pt(2)
        para.add_run(item)

def callout(label, text):
    t = doc.add_table(rows=1, cols=1); t.style = "Table Grid"
    c = t.cell(0,0); shade(c, PALE); c.text = ""
    q = c.paragraphs[0]; q.paragraph_format.space_after = Pt(3)
    a=q.add_run(label + "  "); a.bold=True; a.font.color.rgb=RGBColor.from_string(TEAL)
    q.add_run(text)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)

def page(title, kicker=None):
    if len(doc.paragraphs) or len(doc.tables): doc.add_page_break()
    if kicker:
        x=p(kicker.upper()); x.paragraph_format.space_after=Pt(2)
        x.runs[0].font.size=Pt(8); x.runs[0].font.bold=True; x.runs[0].font.color.rgb=RGBColor.from_string(TEAL)
    p(title, "Heading 1")

# 1
p("MIDORI", "Title")
p("Multimodal Dataset and Model Design", "Heading 1")
p("A practical dossier for a 12-joint quadruped, from logged episodes to phone-based intelligence and safe robot control", "Subtitle")
doc.add_paragraph()
callout("Design objective", "Build a compact multimodal assistant that interprets voice, images, and robot state, then selects safe locomotion skills. Keep fast balance and servo control in a deterministic controller on the robot, rather than asking a language model to drive 12 servos directly at every instant.")
p("Prepared for the Midori quadruped prototype", "Heading 2")
p("Design assumption: a 150-million-parameter model is a target for a later compact system, not evidence that the current data or compute budget is already sufficient. The proposed 5-billion-token budget is an approximate training-plan target. Its usefulness depends on how tokens are counted, modality coverage, compute, and data quality.")
p("What this dossier specifies", "Heading 2")
bullets(["A versioned, timestamp-aligned episode format for text, audio, images/video, robot state, actions, and outcomes.","A repeatable capture, quality-control, split, and loader pipeline that avoids turning every CSV row into wasteful text tokens.","A staged model plan: modality encoders, a shared sequence model, structured skill/action heads, supervised training, simulation reinforcement learning, and guarded deployment.","Concrete recommendations for the existing MuJoCo data and the future phone-to-ESP32 system."])
callout("First priority", "Improve the action and observation contract and collect diverse, synchronized episodes before scaling model size. Ten thousand near-identical samples do not substitute for coverage of commands, terrain, viewpoints, failures, and recovery.")
p("Version 1.0  |  26 October 2026", "Normal")

# 2
page("Executive recommendation", "01 / Summary")
p("Use an episodic dataset and a hierarchical controller. A high-level multimodal policy should turn an instruction plus recent observations into a short, bounded locomotion command. A gait controller should turn that command into phase-coordinated foot paths and 12 joint targets. The ESP32 should enforce limits, watchdogs, and the servo update loop.")
table(["Layer", "Responsibility", "Typical output"],[
["Phone model", "Interpret instruction, image, audio, state; choose skill and intent", "walk forward, vx, yaw rate, duration, confidence"],
["Robot controller", "Gait phase, inverse kinematics, joint limits, state feedback", "12 bounded servo targets at control rate"],
["ESP32 / actuator", "PWM or bus servo timing, watchdog, emergency stop", "physical motor signals"],
["Recorder", "Timestamp and persist observations, commands, results", "episode metadata + Parquet + media"],
])
p("Why this division works", "Heading 2")
p("A transformer can plan and interpret context, but its inference latency and output jitter are a poor foundation for balance-critical inner loops. The robot needs predictable timing even when the phone model is loading an image or decoding speech. Decoupling preserves a safe gait if the model pauses, disconnects, or returns an invalid command.")
p("Architecture choice", "Heading 2")
p("For an initial 150M-parameter system, use modality-specific front ends that convert camera, audio, and numeric robot state into embeddings, followed by a shared decoder-only transformer. Add a small structured action head for skill, velocity, yaw rate, and duration. Do not spend the entire budget on a text decoder while leaving no capacity for image/audio encoders and action prediction.")
callout("Recommendation", "Start with a behavior-cloned high-level policy over a conventional gait controller. Add RL in simulation after the observation/action interface and reward are stable. Test the learned policy as a proposal source behind a command validator.")

# 3
page("System scope and assumptions", "02 / System")
p("The current robot is an approximate quadruped with four legs and three joints per leg: 12 actuated degrees of freedom. The current MuJoCo work uses hello.xml and a Python controller. It is a useful first simulator, but it is not yet a calibrated digital twin of the printed body or the physical servos.")
table(["Known project context", "Implication for dataset/model"],[
["12 joints, three per leg", "Record every joint by stable name, leg, axis, unit, and sign convention."],
["Current motion CSV samples at wall-clock 10 Hz", "At 100 ms intervals, simulation time may advance unevenly; do not assume fixed sim-time spacing."],
["Current 'actual' angles come from MuJoCo state", "Label them simulated observations; they are not physical encoder telemetry."],
["Virtual front camera can save PNG frames", "Images are snapshots unless a scheduled video capture loop is added."],
["Future ESP32 and phone link", "Define a versioned command packet and a local controller independent of the model."],
])
p("Scope boundary", "Heading 2")
p("This plan covers a locomotion-centered multimodal system: commands, speech, camera context, proprioception, gait actions, and episode outcome. It does not assume object manipulation, mapping, navigation, or autonomous obstacle negotiation are solved. Those should become separate task families with explicit sensors and labels.")
p("Important design assumption", "Heading 2")
p("The planned model learns from demonstrations and interaction traces. It does not automatically learn physics simply because data contain positions and forces. Physical consistency should be represented by the simulator, explicit derived quantities, constraint checks, and carefully weighted auxiliary losses. A physics-informed loss is a supplement to correct data and control, not a substitute for them.")
callout("Calibration gate", "Before using real-robot logs for policy training, measure joint direction, neutral pulse, range, speed, chassis dimensions, camera placement, and timing. Store each calibration as a versioned artifact and reference it from every episode.")

# 4
page("What to collect: modality inventory", "03 / Dataset")
p("Collect synchronized episodes, not unrelated files. Each episode is one bounded attempt at a task, from reset or start state through success, failure, or safe abort. Preserve the raw recording, then create normalized derived data without overwriting the source.")
table(["Stream", "Capture", "Purpose / labels"],[
["Instruction", "Typed command; language, locale, task ID", "What the operator asked; normalized intent may be an additional label."],
["Audio", "Original waveform plus sample rate and timestamp", "Speech, prosody/noise robustness; retain transcript and transcription confidence separately."],
["Camera", "Timestamped RGB frames or video; optional depth later", "Scene context, floor, obstacle, visible body posture; camera calibration and frame time."],
["Robot state", "Root pose/velocity, IMU, joint position/velocity/current if available", "What the robot sensed; source must identify simulated vs measured."],
["Action", "Requested skill/velocity and 12 joint targets; actual actuation feedback", "What was commanded and what the robot actually executed."],
["Outcome", "Success, fall, collision, timeout, operator stop, reward components", "Training targets, evaluation grouping, RL reward and safety analysis."],
])
p("Capture conditions to vary", "Heading 2")
bullets(["Command wording, speaker, microphone distance, background noise, and speech rate.","Lighting, camera angle, floor texture, harmless obstacles, and initial body pose.","Forward, backward, turning, stop, slow/fast gait, start/stop transitions, and recovery attempts.","Simulation parameters such as friction and actuator delay; real tests only within safe mechanical limits."])
callout("Do not conflate labels", "Keep target joint position, simulator/measured joint position, requested high-level motion, and applied PWM or bus command as separate fields. They answer different questions and can diverge during saturation or contact.")

# 5
page("Episode design and synchronization", "04 / Capture")
p("Use one monotonic clock as the episode reference. Store wall/UTC time for audit, but align sensor streams by monotonic nanoseconds or a device clock mapped to it. Include both simulation time and real elapsed time when recording MuJoCo. This matters because the present CSV recorder samples on wall-clock intervals while the simulation may run faster or slower.")
p("Episode lifecycle", "Heading 2")
bullets(["Create an episode ID and write metadata before capture: task, robot/simulator build, calibration, operator, and clock source.","Record start state and instruction. Timestamp each observation and each command at acquisition/application time, not when later written to disk.","At end, write terminal status, success/failure reason, operator notes, and final state; flush and validate every file.","Never silently interpolate across a long sensor gap. Mark the gap and make the affected window ineligible for a loss that needs continuous timing."])
table(["Stream", "Suggested rate / handling", "Alignment notes"],[
["Servo/controller state", "Controller-native rate; retain raw rate, derive 10–50 Hz model view", "Use monotonic device timestamp; preserve each original sample."],
["High-level intent", "Event-based", "Record press and release; command has explicit expiry."],
["Camera", "5–15 fps initially, or event snapshots", "Timestamp exposure/capture; do not copy one frame as if it were new."],
["Audio", "Continuous chunks with sample index and time", "Link transcript word/chunk times if available."],
["MuJoCo", "Save sim_time and wall monotonic time", "Resample by sim_time for simulation dynamics analysis; retain wall time for system behavior."],
])
callout("Current logger reading", "The current 10 samples/second setting is a wall-clock sampling interval. A row's sim_time_s is the correct simulation-time coordinate for dynamics, but its spacing may vary. Do not describe this log as fixed 10 Hz simulation data without checking those timestamps.")

# 6
page("Storage layout and data contract", "05 / Storage")
p("Keep the MuJoCo project as one component inside the repository's mujoco folder, and keep the growing dataset in its own data tree. A stable episode contract lets the simulator, future ESP32 logger, and training loader share the same semantics.")
table(["Path", "Contents"],[
["data/raw/", "Immutable original CSV, WAV, image/video, and device logs."],
["data/episodes/<episode_id>/", "episode.json, state_action.parquet, media references, annotations."],
["data/derived/<version>/", "Resampled windows, normalized signals, features, filtered episodes."],
["data/splits/<version>/", "Episode IDs assigned to train, validation, test; split rationale."],
["data/schemas/", "Schema, units, coordinate frames, action vocabulary, migration notes."],
])
p("Core table fields", "Heading 2")
p("Each state/action row should include episode_id, step_index, monotonic_time_ns, sim_time_s or device_time_ns, source (mujoco/hardware), robot_build, calibration_id, movement_mode, task_id, root pose and velocity with coordinate frame, all 12 target angles, all 12 observed angles and velocities, optional current/temperature/contact, four foot positions if available, command validity, and terminal/event flags. Use explicit SI units in names or schema metadata.")
p("Media manifest", "Heading 2")
p("Store camera frames/video and audio as files; put their paths, timestamp ranges, codec/sample rate, dimensions, and checksums in a manifest. Keep waveform samples and image pixels out of a giant CSV. Parquet is a practical choice for numeric columns and columnar scans; CSV can remain a raw interchange/export format. RLDS is a useful reference for episode/step organization, not a mandatory dependency.")
callout("Versioning", "Version the schema and robot calibration independently. A changed joint sign or coordinate frame is a semantic breaking change even when the CSV columns still look identical.")

# 7
page("Quality checks, labels, and splits", "06 / Data quality")
p("A clean loader cannot recover a mislabeled action. Run checks at ingest and store their results beside the episode. Reject corrupt files, quarantine uncertain episodes, and keep provenance so a model result can be traced back to a capture session.")
table(["Check", "What it catches"],[
["Schema and units", "Missing columns, degree/radian mixups, wrong axis or joint ordering."],
["Clock integrity", "Non-monotonic timestamps, duplicate samples, camera/audio drift, long gaps."],
["Physical bounds", "Joint limits, impossible velocities, NaN/Inf, chassis teleportation, invalid quaternion norm."],
["Action consistency", "Direction key and gait label mismatch; action expiry missing; target vs actual swapped."],
["Media integrity", "Unreadable frames, empty audio, wrong dimensions/rate, broken manifest paths."],
["Episode outcome", "Unknown terminal reason, success without criterion, unsafe abort not marked."],
])
p("Labels and uncertainty", "Heading 2")
p("Keep original operator text and speech transcript. Any normalized intent is another field, not a replacement. Store label source (human, script, simulator), confidence, annotator, and revision. A successful trajectory should state its success criterion; a stumble or fall should remain a useful negative example when the robot was stopped safely.")
p("Split without leakage", "Heading 2")
p("Assign whole episodes—and preferably whole capture sessions or environment variants—to one split. Never randomly split adjacent rows from one motion into train and test: nearly identical frames and gait cycles make test scores misleading. Keep a final held-out test set untouched during tuning, and publish its task/environment composition with each evaluation.")
callout("Minimum dashboard", "Track episode count and duration; valid synchronized duration; task and outcome balance; missingness by sensor; joint-bound violations; success and fall rates; unique speakers/scenes; and train/validation/test coverage. Report raw and filtered counts.")

# 8
page("How raw data becomes model examples", "07 / Loader")
p("The model consumes aligned windows, not one training example per CSV line. For each sampled time t, construct a short past context and a future action target. The context may contain an instruction, recent audio/transcript, a few recent camera frames, and a compact state history. The target is an action chunk over the next bounded horizon.")
table(["Example part", "Representation"],[
["Text", "Tokenizer IDs with task and instruction markers."],
["Audio", "Waveform or log-mel/features into an audio encoder; optional transcript tokens."],
["Image/video", "Selected frames through a vision encoder; timestamp and camera ID retained."],
["Numeric state", "Normalized continuous vector plus validity mask and time delta."],
["Action target", "Categorical skill plus continuous vx, vy, yaw rate, duration, or 12-joint chunk for specialist tasks."],
])
p("Conceptual training sample", "Heading 2")
callout("Input", "<instruction> Walk slowly toward the open area. <image t=-0.4s> [frame embedding] <state t=-0.1s> [body pose, joint state, validity mask] <audio> [optional speech embedding] <state now> [...]")
callout("Target", "<skill> WALK <vx> 0.10 m/s <vy> 0 <yaw_rate> 0 <duration> 200 ms <stop_if> obstacle_or_timeout")
p("Loss masks", "Heading 2")
p("Only score the output fields that are labels for that task. Do not train the model to reproduce every historical input token as if it were a desired action. Use separate objectives: language/audio/image representation tasks where data support them, behavior cloning on action targets, and optional state/dynamics prediction. Attach a time mask to missing modalities rather than filling them with fabricated values.")
p("Windowing", "Heading 2")
p("Begin with 1–3 seconds of context and 0.2–0.5 seconds of high-level action horizon, then measure latency and control smoothness. Stride windows so adjacent examples are not mistakenly counted as independent episodes. Downsample camera frames intentionally; preserve motion-relevant frames and timestamps.")

# 9
page("Tokens, storage, and the 5B budget", "08 / Scale")
p("A language tokenizer's token count is not a universal unit for image pixels, waveform samples, or robot state. Track at least three budgets separately: text tokens, audio seconds/encoded frames, image frames/patches, and robot transitions or episode-hours. If a single scalar ‘multimodal token’ total is reported, document the encoder and conversion rule.")
p("Why CSV row count is not token count", "Heading 2")
p("The current logger writes about 10 rows per wall-clock second. One hundred thousand rows therefore represent about 2.78 hours of wall-sampled logging, assuming uninterrupted recording. That is not 5 billion language tokens, nor 5 billion independent decisions. Simulation-time duration may differ because the logger's sampling clock is real time while MuJoCo can run at another speed.")
table(["Budget item", "Measure separately"],[
["Text pretraining", "Tokenizer version, non-padding tokens, source and deduped tokens."],
["Audio", "Hours, sample rate, language/speaker coverage, encoded sequence length."],
["Vision", "Frames, resolution, scenes, camera/viewpoint, patch/token cost."],
["Robot experience", "Episodes, valid hours, transitions, action horizon, success/failure balance."],
["Training compute", "Accelerator-hours, throughput, wall time, energy/cost, checkpoint cadence."],
])
p("Practical implication", "Heading 2")
p("A 150M model trained on 5B text-like tokens is a different plan from training a multimodal policy on 5B mixed encoded units. Count tokens after the actual tokenizer/encoders and after masking. Do not pad the corpus with duplicated or low-value web text to hit a headline number. A compact robot policy may gain more from clean, diverse, aligned robot episodes and robust audio/vision representations than from arbitrary extra tokens.")
callout("Recommendation", "Publish a data manifest with raw hours, episode counts, usable aligned duration, and per-modality compute tokens. Keep the 5B target provisional until tokenizer, encoder, context length, hardware, and training schedule are chosen.")

# 10
page("Recommended model architecture", "09 / Model")
p("A practical first design is a multimodal encoder plus a shared autoregressive transformer and a typed action head. “Decoder-only” describes the shared sequence model; it does not require raw pixels, audio samples, and 12 floating-point angles to be serialized as ordinary text.")
table(["Module", "Role", "Design note"],[
["Text tokenizer + embeddings", "Commands, transcripts, task markers", "Small vocabulary suitable for target languages; keep raw transcript."],
["Vision encoder + projector", "Frame/patch features into shared width", "Use low resolution or few frames first; include timestamps."],
["Audio encoder + projector", "Speech/audio features into shared width", "Support direct audio instruction and optional transcript supervision."],
["State encoder", "Numeric values, masks, time deltas", "Normalize by calibration/statistics; embed joint identities/leg IDs."],
["Shared transformer", "Fuse temporal context and instruction", "Causal sequence model; modality/type and time embeddings."],
["Action heads", "Skill, velocity, yaw, duration, stop/confidence", "Categorical + continuous distributions; bounded decoding."],
])
p("A cautious 150M allocation", "Heading 2")
p("Treat 150M as the total trainable parameter ceiling, not just the language backbone. A starting budget might reserve 90–110M for the shared transformer, 20–35M for visual/audio/state adapters or compact encoders, and 5–15M for output heads and embeddings; exact counts depend on tying, vocabulary, and whether modality encoders are trained from scratch. Verify with a parameter-count script before committing to training.")
p("Alternative", "Heading 2")
p("A unified discrete-token approach can model interleaved text and modality tokens, as explored by early-fusion work. For this project, continuous state and actions are naturally structured, and sensor rates differ. Modality encoders with explicit timestamps are the more manageable first system; consider more aggressive unified tokenization only after reliable data pipelines exist.")

# 11
page("Action space and walking controller", "10 / Control")
p("Make the model choose intent and let the locomotion controller execute the gait. Do not use keyboard labels such as “forward” as if they were a complete action: speed, yaw, duration, gait state, and stop behavior must be explicit. The current controller already has a crawl gait/IK path; this can form the deterministic baseline and action decoder.")
table(["Model action", "Controller meaning", "Guard"],[
["skill = WALK", "Continue crawl gait while command is valid", "Only allow known skills."],
["vx, vy", "Desired body translation in m/s", "Clamp to tested range; reject NaN."],
["yaw_rate", "Desired turn rate in rad/s", "Clamp; ensure left/right sign is calibrated."],
["duration_ms", "Short action chunk before replanning", "Hard maximum and automatic expiry."],
["stop / confidence", "Stop intent or uncertainty signal", "Unknown/expired command defaults to safe stop."],
])
p("Why not direct 12-joint language output first?", "Heading 2")
p("Direct joint prediction makes the model responsible for balance, contact timing, actuator limits, and timing jitter at once. A useful later experiment is a joint-action chunk head, trained from logged targets, while the gait controller remains an independent fallback. Compare it against the high-level skill policy under the same held-out terrain and perturbation tests.")
p("Servo command bridge", "Heading 2")
p("The phone sends a compact versioned packet over a local link, for example: sequence number, skill ID, vx, vy, yaw rate, duration, confidence, CRC. The ESP32 checks version, range, freshness, and watchdog; then the gait controller maps motion to 12 joint targets. Log both received intent and actual servo target/feedback. The ESP32 should be able to stop without waiting for the phone.")
callout("Safety invariant", "Any lost connection, stale packet, invalid field, model timeout, or emergency-stop input must cause a defined controlled stop. Validate this behavior without model inference first.")

# 12
page("Training plan: pretraining and alignment", "11 / Training")
p("Training from scratch is a multi-stage program, not one run over a merged folder. Keep a baseline for every stage and freeze a reproducible dataset manifest, code revision, tokenizer/encoder version, configuration, and seed.")
table(["Stage", "Data and objective", "Exit evidence"],[
["A. Data plumbing", "Schema validation, media decoder, timestamp alignment, episode windows", "Reproducible batches; no silent sensor/label swaps."],
["B. Modality foundations", "Text next-token; audio tasks; image/scene tasks; state normalization", "Each encoder learns useful representations on held-out examples."],
["C. Cross-modal alignment", "Matched instruction/audio/transcript/frames/state; contrastive or paired prediction", "Correct matching beats shuffled mismatches; ablation sanity checks."],
["D. Joint pretraining", "Interleaved multimodal context; masked/predictive tasks where labels exist", "Stable loss curves and modality-specific validation metrics."],
["E. Behavior cloning", "Observation/instruction -> bounded skill/action chunks", "Held-out command and environment performance; stop behavior verified."],
["F. RL in simulation", "Policy improvement with structured rewards and safety constraints", "Improved task score without worsening falls, collisions, or limits."],
])
p("Balance the mixture deliberately", "Heading 2")
p("Set sampling rates by task utility and gradient contribution, not file size. If text dominates batches, the robot policy can remain poor despite a strong text loss. Track per-task losses and sampling ratios. Use data deduplication and episode-aware sampling; upweight rare recovery and stop cases carefully without fabricating success labels.")
p("From-scratch realism", "Heading 2")
p("A 150M model can be trained from scratch, but compute and dataset quality remain decisive. First run small-scale pilots, estimate throughput and validation curves, then decide whether all modality encoders should be trained from scratch or whether a permitted pretrained encoder is a better engineering choice. If ‘from scratch’ is a firm requirement, budget the additional data and compute explicitly.")

# 13
page("SFT, reinforcement learning, and physics", "12 / Learning")
p("Supervised fine-tuning (SFT) should teach the system to follow instructions and imitate safe demonstrations. Use explicit task labels and action targets. Include “do nothing/stop,” cancellation, partial completion, and uncertainty cases. SFT quality is bounded by demonstration coverage: it cannot infer a safe gait for terrain or commands absent from the data without generalization evidence.")
p("RL sequence", "Heading 2")
bullets(["Begin in MuJoCo with resets, randomized friction/mass/delay, diverse starts, and domain-randomized camera conditions.","Use reward components for commanded velocity tracking, heading error, body stability, foot clearance/contact, energy, joint limits, collision, and fall termination.","Log reward components and policy/version, not just total reward. Use curriculum from standing and short moves to turns, transitions, and perturbation recovery.","Test learned policies against fixed baselines and hold out simulator parameter ranges; then cautiously evaluate on a tethered/guarded physical robot."])
p("Physics-informed components", "Heading 2")
p("For this robot, useful physics losses can penalize impossible state changes, violated joint bounds, inconsistent forward kinematics, or predicted dynamics residuals when the necessary masses, inertias, and contacts are known. A learned dynamics model can support short-horizon prediction or model-based planning. Keep these auxiliary losses separate and report ablations; contact-rich locomotion is hybrid and noisy, so an elegant equation does not guarantee robust ground truth.")
table(["Learning signal", "What it teaches", "Caution"],[
["Behavior cloning", "Copy safe demonstrated action", "Can reproduce demonstrator bias and failures."],
["RL reward", "Improve task outcomes through interaction", "Reward hacking and simulator mismatch."],
["Physics residual", "Respect selected kinematic/dynamic constraints", "Wrong parameters produce confidently wrong gradients."],
["World/state prediction", "Anticipate near-future body/scene change", "Prediction quality is not the same as safe control."],
])

# 14
page("Evaluation, deployment, and roadmap", "13 / Delivery")
p("Evaluate the complete phone–robot loop, not just language accuracy. Use frozen held-out episodes and repeatable real-robot protocols. Measure both task success and how often the safety layer intervenes.")
table(["Area", "Metrics"],[
["Instruction/audio", "Intent accuracy, transcription/command error, rejection of ambiguous command."],
["Locomotion", "Velocity/yaw tracking, command latency, gait smoothness, falls, collisions, recovery."],
["Control safety", "Joint-limit violations, stale-packet stop time, watchdog response, emergency-stop success."],
["Multimodal grounding", "Correct scene/obstacle interpretation, performance with a missing modality."],
["Runtime", "Phone memory, model latency, power/thermal behavior, link loss and reconnect behavior."],
])
p("Suggested roadmap", "Heading 2")
bullets(["1. Stabilize the simulator and record explicit action press/release, timestamps, source type, and episode outcomes.","2. Add a versioned state/action/media schema and one validated loader; convert logs without discarding raw CSV.","3. Collect a small, diverse demonstration set and build a conventional gait controller with a command expiry/watchdog.","4. Train baseline intent and behavior-cloning models; evaluate held-out episodes and compare with scripted control.","5. Add synchronized audio and camera windows, then repeat ablations and latency measurements.","6. Add simulation RL and physics-aware auxiliary objectives only after the data/action contract is stable.","7. Integrate an on-phone quantized model; keep the ESP32 safety controller authoritative. Run guarded hardware trials and update calibration."])
callout("Deployment boundary", "A 150M model may fit on a capable mobile device after quantization, but actual memory, latency, thermal limits, and accelerator support must be measured on the target phone. Keep a safe non-ML gait fallback on the robot.")

# 15
page("Scale, risks, and immediate decisions", "14 / Close")
p("The near-term bottleneck is data quality and control semantics, not the headline token count. The current logger is useful for simulator trajectories and bounded CSV files, but it does not yet provide synchronized audio/video or physical servo feedback. Treat those as separate engineering milestones.")
table(["Risk", "Mitigation"],[
["Simulation-to-real gap", "Record hardware telemetry, calibrate, randomize sim, use staged guarded trials."],
["Misaligned streams", "Monotonic clocks, timestamped media, explicit gaps and drift dashboards."],
["Unsafe generated actions", "Typed bounded actions, command expiry, watchdog, stop fallback, range validation."],
["Overstated dataset scale", "Report episode-hours and valid transitions as well as modality token counts."],
["Model too large for phone", "Profile target device early; quantize and measure; reduce context/encoder size as needed."],
["“Physics informed” used vaguely", "Name each constraint, units, applicable state, coefficient, and ablation result."],
])
p("Immediate decisions to lock", "Heading 2")
bullets(["Define the robot coordinate frame, joint ordering/signs, units, neutral pose, and safe motion envelope.","Choose the first task: command-conditioned slow crawl on a known floor, with stop and turn included.","Set the episode schema, timestamp strategy, action packet version, and success/failure labels.","Choose target phone, ESP32 communication link, servo feedback availability, and required offline behavior.","Estimate training compute before committing to 5B tokens; distinguish text tokens from audio/image/state sequences."])
p("References and further reading", "Heading 2")
p("Open X-Embodiment (Google DeepMind), episode-based RLDS data organization: https://github.com/google-deepmind/open_x_embodiment", "Normal")
p("RT-2, vision-language-action policy and discretized action outputs: https://arxiv.org/abs/2307.15818", "Normal")
p("Qwen2-Audio Technical Report, audio encoder plus language-model approach: https://arxiv.org/abs/2407.10759", "Normal")
p("Chameleon, mixed-modal early-fusion foundation models: https://arxiv.org/abs/2405.09818", "Normal")
p("LeRobot dataset v3 documentation, episode/task-oriented robotics dataset tooling: https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx", "Normal")
callout("Final recommendation", "Build the dataset around synchronized episodes and learn a safe hierarchical policy. Keep low-level locomotion deterministic and measurable while the multimodal model learns when and how to request movement.")

doc.core_properties.title = "Midori Multimodal Dataset and Model Design"
doc.core_properties.subject = "15-page design dossier for a multimodal quadruped robot learning system"
doc.core_properties.author = "Midori Project"
doc.save(OUT)
print(OUT)
