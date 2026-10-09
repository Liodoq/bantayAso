# BantayAso motion and behavior training plan

Status: proposed next work after the approved six-issue repair. No new model weights have been trained. Written Oct 9, 2026; operational details and completed checks are in ../BRIEF.md. All quantities below are starting targets, not measured performance or guarantees.

## Recommendation

### Owner recording checklist (Oct 9 follow-up)

**Record video as the primary material.** Still photos can help posture/identity, but they cannot show the order of movements. We can extract posture images from the recordings later. Use Bantay's Record clip with the intended monitoring camera, camera position and normal resolution; it saves clean footage without the app overlay. Keep the camera fixed during each take, with head, paws and whole body visible where possible.

- Start with 10–20-second clips containing a clearly visible 5–10-second action interval. Leave a few seconds before/after a transition. Do not force the dog to hold an uncomfortable pose.
- Record sitting, lying awake, standing, walking, licking, scratching and safe chewing/eating. Include ordinary toy chewing and calm/no-dog scenes. Label sleeping only when observable; otherwise use resting/lying.
- For a first collection session, aim for 3–5 independent takes per core behavior across available dogs. This is a pilot for annotation and diagnosis, not enough to claim a trained/generalized model. Expand toward the independent-episode targets below once gaps are clear.
- Vary side/front views, distance, lighting and sessions. Do not rotate the camera continuously during a take; record different views as separate takes. Use the deployment view for the majority of clips.
- After Stop recording, fill in the new labeling form: dog name(s), behavior and notes/times. For existing or locally copied phone videos, use Menu → Label a recorded clip. Labels are saved beside each video as `<video filename>.labels.json`; these are clip summaries, not machine-parsed interval annotations yet. Include the session and exact start/end seconds in notes, including uncertainty and overlapping behaviors such as lying + chewing. Example: `clip.mp4; Oreo; evening-01; sitting 2–8 s; standing 8–11 s; walking 11–15 s`.
- Reserve entire fresh sessions for validation/test; never split adjacent frames from the same take between training and testing. Grouped splitting supports keeping related samples together ([scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)).

Recording does not automatically train Bantay. The next steps are annotation, feature/model training, validation and deploying a versioned model. Keep clips and learned arrays backed up locally: `data/` and `models/` are gitignored, so pushing the code to GitHub does not back them up.

High-level path: **stabilize capture/motion → label representative videos → measure a baseline → train and compare → validate live operation → release with rollback**. For the app, include startup/device diagnostics, camera reconnect, save/reload, clear unknown/error states, voice/DND behavior, bounded memory/queues and a 30-minute soak check. Keep perception uncertainty visible instead of translating every weak signal into a confident behavior.

### Model direction

Keep the working dog/hazard detectors initially. Fix measurement and build a labeled video evaluation set before buying accuracy with a larger model. Use the existing CLIP encoder as a frozen feature extractor, then compare a small supervised classifier and a small temporal classifier on the same held-out sessions. Separate posture, movement and mouth activity: a dog can lie down AND chew.

The six repaired software defects do not make the underlying behavior model accurate. The current system classifies individual pictures and averages their scores; it does not learn the order of movements. Voice teaching currently stores six embeddings, not a video, not six independent episodes, and not new model weights.

## 1. Fix the input measurements first

Evidence from the current `MotionMeter` on controlled synthetic images:

- Same static image, dog box shifted by five pixels: energy **25.238**, reported **frantic**. Box movement alone created the apparent motion.
- Same textured subject translated across the frame, with a perfectly following box: energy **0.0**, reported **still**. Resizing/cropping removed the whole-body movement.
- These diagnostics isolate algorithm limitations. They are not real-dog error rates.

Proposed changes, in order:

1. Keep timestamps and box centers, sizes, confidence, observed-vs-held state and track IDs. Estimate translation in body-lengths per second, compensating for camera motion; do not treat it as the same measurement as movement within the body.
2. Stabilize the comparison region. Align consecutive dog crops, mask background where possible, and reset motion history after unreliable alignment, severe box jumps, occlusion or track changes. A blank/held crop should produce **unknown**, not still.
3. Compare aligned optical flow with aligned frame differences. Track residual movement distribution in the head, torso and legs separately where those regions are reliably visible. Without pose landmarks, mark approximate head regions as uncertain.
4. Estimate camera movement from background features. Add tests for camera shake, lighting change and compression noise. Do not compensate away genuine walking when removing camera motion.
5. Add confidence/unknown states. Avoid turning one uncertain “scratching” prediction into a confident sitting/eating label while retaining the old label's score. This is an additional issue in the current motion fallback to address in this phase.

OpenCV provides optical-flow and image-alignment primitives; this proposal still needs a comparison on Bantay's footage. See [OpenCV tracking and alignment documentation](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html). Check the installed OpenCV 5.0 bindings before using a particular API.

**Exit check:** stationary subject + box jitter no longer reports frantic; translated subject registers movement; camera shake/occlusion produces compensated motion or unknown. Then check these behaviors on real clips at multiple frame rates and distances.

## 2. Define labels that match what a camera can observe

Use multiple label groups instead of one mutually exclusive list:

- **Posture:** sitting, standing, lying, unclear. Keep “sleeping” separate only when the eye/head view supports it; otherwise say resting/lying.
- **Movement:** stationary, walking, running, jumping, digging, scratching, unclear.
- **Mouth/self-care:** chewing/eating, licking body/paw, sniffing, panting, none visible, unclear. Distinguish food eating from other chewing through visible objects/zones only when supported.
- **Interaction:** alone, calm contact, active play, uncertain rough interaction. Do not promise to infer aggression from overlap or a still-image similarity score; use “rough interaction—check” until separately validated.
- **Context:** named object, object tier, zone, identity confidence, visibility and camera condition. Risk remains a separate rule decision.

First target the recurring confusions: lying vs sitting, scratching vs sitting, licking vs chewing, walking vs standing. Expand only after these are measured. Keep labels and their definitions in a versioned annotation guide with positive, negative and ambiguous examples.

## 3. Build a small, honest dataset

Start with existing clips rather than staging everything again:

- `rec_20261009_175137`: historical calm/empty development clip.
- `rec_20261009_173621`: historical food/eating calibration clip.
- `rec_20261009_174254`: brief wrapper pickup and chewing, according to the earlier handoff.
- `rec_20261009_174656`: multi-dog paper chewing/shredding and overlap, according to the earlier handoff.

These descriptions are leads for manual annotation, not ground truth. All four clips have already influenced development, so use them as development/regression footage, **not** a clean final test set. The new GPU smoke test sampled 0/1/2 seconds of 175137 and verified inference only.

Collect a starter set of approximately **20–30 independent 5–10 second episodes per core behavior**, across the dogs, different orientations, near/far views, multiple sessions and normal lighting conditions. That is a data-collection target; collect more based on error analysis rather than claiming a fixed sample count ensures accuracy. Record longer calm sessions too, because false alarms per hour cannot be evaluated on short action clips alone.

Include hard negatives: licking, panting, scratching, yawning, head turns, ordinary toy chewing, grooming while lying down, a person walking past, overlapping dogs, empty scenes, camera shake and partially hidden heads. Capture naturally occurring actions; do not induce aggression or expose dogs to dangerous objects. Use safe props with access controlled when evaluating hazard rules.

Store an annotation manifest with recording/session ID, file, start/end time, dog ID, track/bounding-box reference, each label group, visibility/occlusion, zone/object context, annotator confidence and split. Mark uncertain examples and exclude them from confident supervised labels; retain them for abstention tests. Keep original clips immutable and video/model files outside Git.

Split **by recording session/episode before generating crops or overlapping windows**. A proposed starting split is 60% training, 20% validation, 20% test, stratified where possible. Never put nearby frames or windows from the same episode into different splits. Add a separate held-out-dog or room/camera-angle check when the dataset can support it. If a class appears in only one session, collect more sessions before claiming generalization.

## 4. Compare increasingly capable models

1. **Baseline:** current prompts and repaired rule pipeline, without taught examples. Save predictions and errors against the labeled development set.
2. **Frozen-image baseline:** cache normalized CLIP features once; train a regularized linear classifier for each label group. Compare it with the conservative example matcher. This isolates whether better labels alone help before adding temporal complexity.
3. **Recommended temporal candidate:** use a short causal window, initially **8–16 observations over 2–4 seconds**, with timestamps, frozen CLIP features and corrected motion/trajectory features. Train a small 1-D temporal convolution or GRU head; compare against simple mean/max pooling. Use only past frames, handle missing observations explicitly, and reset state across identity changes. Window length is a validation choice; a four-second window need not delay immediate hazard rules.
4. **Later candidates only if needed:** fine-tune an efficient video backbone or add dog-specific pose landmarks when failures show that the frozen features/head region are insufficient. TSM is a published example of adding temporal processing efficiently; it is a candidate family, not an already compatible replacement or proof of dog accuracy. See [TSM paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html) and [authors' implementation](https://github.com/mit-han-lab/temporal-shift-module).

The broader [Animal Kingdom dataset paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ng_Animal_Kingdom_A_Large_and_Diverse_Dataset_for_Animal_Behavior_CVPR_2022_paper.html) provides an animal-video action/pose research reference. Inspect relevant dog classes, labels, licensing and domain match before using it; generic animal behavior data does not replace footage from this camera and these dogs.

On the RTX 3050's 4 GB VRAM, cache embeddings in batches and train the small head separately from live monitoring/Ollama. Measure memory and latency before loading another backbone. No automatic downloads, GPU-capacity assumptions, cloud training or OS memory changes are part of this plan.

## 5. Evaluate behavior quality and alert quality separately

- Per-class precision/recall and macro F1; confusion matrices for posture/mouth/motion groups.
- Event-level detection rate, onset delay, duration errors and false Warning/Danger alerts per monitored hour. Report configured persistence time alongside delay.
- Unknown/abstention rate and coverage. Low false alarms achieved by saying unknown constantly is not a useful improvement.
- Identity swaps, track fragmentation, missed dogs and failures caused by small/occluded crops; do not blame the action head for an absent dog crop.
- End-to-end FPS, median/p95 latency, peak memory and a 30-minute soak test on the laptop. Record dog count, source resolution and enabled models with every result.

Proposed release goals for the pilot: a material reduction in calm-scene false alerts (aim for at least 50% against the baseline), no measured loss of hazardous-chewing event recall on the same held-out cases, at least 15 FPS on the intended demo configuration, and no restart over the soak run. These are **targets**, not achieved results. Include sample counts and uncertainty; a handful of successful clips cannot establish reliable recall. Keep direct high-tier hazard rules independent of the temporal action model.

Fit confidence/unknown thresholds and example-matching limits on **validation sessions**. Freeze them before running the untouched test set. The repair's leave-one-out reference gates are a conservative safeguard; they still require this held-out validation and may reject many unfamiliar views.

## 6. Improve teaching and release safely

The repaired six-frame teaching flow is useful for immediate corrections but is not the final dataset builder. Add a future “record a lesson” flow that captures a short clip, shows the selected dog/action, and lets the owner confirm or discard it. Save session/source/model metadata and enable removing a bad lesson. Avoid automatically trusting every spoken label or continuing a lesson through an identity swap.

Keep a versioned bundle containing encoder/preprocessing versions, label vocabulary, feature/head weights, matching/unknown thresholds, split manifest and evaluation report. Compare the candidate against the current system on the same test footage, then trial it in shadow mode before enabling its alerts. Preserve a rollback path.

## Immediate next session

1. Verify the repaired app with the real camera and voice; neither was accessible from the agent's execution context during the repair.
2. Label a short set of the four existing development clips and record fresh calm/chewing/self-care footage for validation/test.
3. Implement the motion measurement fixes and unknown handling, then compare the baseline on those labels.
4. Collect varied lessons, train the frozen-feature baseline, and only then evaluate the small temporal head.

Update BRIEF.md after each step with actual dataset counts, split IDs, checks, model versions and remaining mistakes. Do not mark training complete merely because a model loads or the code tests pass.
