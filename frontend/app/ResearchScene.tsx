"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import type { ResearchCounts, ResearchStage } from "./api";

/**
 * The desk the whole app is themed after: a warm amber microcomputer under a
 * study lamp, at night. Research progress is projected onto the scene rather
 * than simulated: source orbs, transcript threads, knowledge atoms, and the
 * cluster constellation each appear as their pipeline stage completes.
 */

const MAX_SOURCES = 12;
const MAX_ATOMS = 120;

// Orbits are flattened in Z so nothing swings into the camera's face.
const DEPTH_SQUASH = 0.3;

// Clustered atoms settle onto a camera-facing ring, which reads as a
// constellation rather than two blobs either side of the machine.
const CLUSTER_ANCHORS = 7;
const CLUSTER_RING_X = 1.55;
const CLUSTER_RING_Y = 0.6;
const CLUSTER_RING_Z = -0.35;

const STAGE_INDEX: Record<ResearchStage, number> = {
  search: 0,
  transcribe: 1,
  extract: 2,
  cluster: 3,
  synthesize: 4,
  done: 5,
};

const AMBER = 0xffb257;
const AMBER_DIM = 0xd98a3c;
const SKY_COOL = 0x7fa8ff;

type SceneProps = {
  stage: ResearchStage;
  counts: ResearchCounts;
  finished: boolean;
};

function damp(current: number, target: number, lambda: number, dt: number) {
  return current + (target - current) * (1 - Math.exp(-lambda * dt));
}

export default function ResearchScene({ stage, counts, finished }: SceneProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  // The render loop reads live props without being torn down on every update.
  const propsRef = useRef<SceneProps>({ stage, counts, finished });
  propsRef.current = { stage, counts, finished };

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      return; // No WebGL: the HUD alone still communicates progress.
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.6));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.92;
    renderer.setClearColor(0x05090f, 0);
    renderer.domElement.setAttribute("aria-hidden", "true");
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    // Fog is tinted to the sky's horizon so the meadow dissolves into it
    // instead of ending on a hard line. Stars opt out of fog entirely.
    scene.fog = new THREE.FogExp2(0x1b2740, 0.05);

    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    camera.position.set(0, 1.9, 9.6);
    camera.lookAt(0, 1.05, 0);

    const disposables: Array<{ dispose: () => void }> = [];
    const track = <T extends { dispose: () => void }>(item: T) => {
      disposables.push(item);
      return item;
    };

    /** Soft radial sprite texture, used for light pools and horizon haze. */
    const radialTexture = (r: number, g: number, b: number) => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 128;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
        gradient.addColorStop(0, `rgba(${r},${g},${b},1)`);
        gradient.addColorStop(0.45, `rgba(${r},${g},${b},0.32)`);
        gradient.addColorStop(1, `rgba(${r},${g},${b},0)`);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 128, 128);
      }
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      return track(texture);
    };

    // ---------- Night sky ----------
    // A gradient dome so the frame never falls to flat black behind the desk.
    const sky = new THREE.Mesh(
      track(new THREE.SphereGeometry(120, 32, 20)),
      track(
        new THREE.ShaderMaterial({
          side: THREE.BackSide,
          depthWrite: false,
          uniforms: {
            zenith: { value: new THREE.Color(0x04060f) },
            middle: { value: new THREE.Color(0x0d1b33) },
            horizon: { value: new THREE.Color(0x24344a) },
          },
          vertexShader: `
            varying float vHeight;
            void main() {
              vec4 world = modelMatrix * vec4(position, 1.0);
              vHeight = normalize(world.xyz).y;
              gl_Position = projectionMatrix * viewMatrix * world;
            }
          `,
          fragmentShader: `
            uniform vec3 zenith;
            uniform vec3 middle;
            uniform vec3 horizon;
            varying float vHeight;
            void main() {
              float h = clamp(vHeight * 0.5 + 0.5, 0.0, 1.0);
              vec3 color = mix(horizon, middle, smoothstep(0.42, 0.58, h));
              color = mix(color, zenith, smoothstep(0.58, 0.96, h));
              gl_FragColor = vec4(color, 1.0);
            }
          `,
        }),
      ),
    );
    scene.add(sky);

    // Warm haze sitting on the horizon line, echoing distant light.
    const horizonHaze = new THREE.Sprite(
      track(
        new THREE.SpriteMaterial({
          map: radialTexture(255, 176, 108),
          transparent: true,
          opacity: 0.2,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        }),
      ),
    );
    horizonHaze.scale.set(58, 14, 1);
    horizonHaze.position.set(0, 1, -20);
    scene.add(horizonHaze);

    // ---------- Lighting ----------
    // Night ambience, a warm lamp as the key light, cool moonlight as rim.
    scene.add(new THREE.HemisphereLight(0x3a5583, 0x0b0f18, 0.85));

    const moonlight = new THREE.DirectionalLight(SKY_COOL, 0.8);
    moonlight.position.set(-5, 6, -2.5);
    scene.add(moonlight);

    const fill = new THREE.DirectionalLight(0x9db9ff, 0.32);
    fill.position.set(3, 2.4, 4.5);
    scene.add(fill);

    const screenLight = new THREE.PointLight(AMBER, 1.7, 5.5, 2);
    screenLight.position.set(0, 1.2, 0.7);
    scene.add(screenLight);

    const lampLight = new THREE.SpotLight(0xffcf9c, 30, 7, 0.9, 0.45, 2);
    lampLight.position.set(-1.15, 1.74, 0.5);
    lampLight.target.position.set(0.05, 0.9, 0.3);
    scene.add(lampLight, lampLight.target);

    // Soft warm bounce from the viewer's side so the beige case reads as a form.
    const frontWarm = new THREE.PointLight(0xffd6a4, 1.3, 9, 2);
    frontWarm.position.set(1.1, 1.6, 3);
    scene.add(frontWarm);

    // ---------- Desk ----------
    const woodMaterial = track(
      new THREE.MeshStandardMaterial({
        color: 0x4a3324,
        roughness: 0.72,
        metalness: 0.06,
      }),
    );
    const desk = new THREE.Mesh(
      track(new THREE.BoxGeometry(6.4, 0.16, 2.5)),
      woodMaterial,
    );
    desk.position.y = 0.54;
    scene.add(desk);

    const deskEdge = new THREE.Mesh(
      track(new THREE.BoxGeometry(6.4, 0.42, 0.12)),
      woodMaterial,
    );
    deskEdge.position.set(0, 0.32, 1.19);
    scene.add(deskEdge);

    // A warm pool of lamp light on the desk surface.
    const deskPool = new THREE.Mesh(
      track(new THREE.PlaneGeometry(3.4, 1.7)),
      track(
        new THREE.MeshBasicMaterial({
          map: radialTexture(255, 190, 120),
          transparent: true,
          opacity: 0.3,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        }),
      ),
    );
    deskPool.rotation.x = -Math.PI / 2;
    deskPool.position.set(-0.25, 0.625, 0.2);
    scene.add(deskPool);

    // ---------- Meadow ----------
    const meadow = new THREE.Mesh(
      track(new THREE.PlaneGeometry(140, 140)),
      track(
        new THREE.MeshStandardMaterial({
          color: 0x16241d,
          roughness: 1,
          metalness: 0,
        }),
      ),
    );
    meadow.rotation.x = -Math.PI / 2;
    meadow.position.y = -0.02;
    scene.add(meadow);

    // Wildflowers: the glowing field from the hero art, kept sparse and low.
    const flowerCount = 260;
    const flowers = new THREE.InstancedMesh(
      track(new THREE.SphereGeometry(0.028, 6, 6)),
      track(
        new THREE.MeshBasicMaterial({
          color: 0xffc46a,
          transparent: true,
          opacity: 0.85,
        }),
      ),
      flowerCount,
    );
    const flowerMatrix = new THREE.Matrix4();
    for (let i = 0; i < flowerCount; i += 1) {
      // Foreground band only, so flowers never crowd the desk silhouette.
      const x = (Math.random() - 0.5) * 22;
      const z = 1.9 + Math.random() * 5.4;
      const y = 0.03 + Math.random() * 0.16;
      flowerMatrix.makeTranslation(x, y, z);
      flowers.setMatrixAt(i, flowerMatrix);
    }
    flowers.instanceMatrix.needsUpdate = true;
    scene.add(flowers);

    // ---------- Microcomputer ----------
    const computer = new THREE.Group();
    computer.position.set(0, 0.62, 0);
    computer.scale.setScalar(0.86);
    scene.add(computer);

    const caseMaterial = track(
      new THREE.MeshStandardMaterial({
        color: 0xd8cdb4,
        roughness: 0.62,
        metalness: 0.04,
      }),
    );

    const body = new THREE.Mesh(
      track(new THREE.BoxGeometry(1.16, 1.08, 0.92)),
      caseMaterial,
    );
    body.position.y = 0.54;
    computer.add(body);

    const bezel = new THREE.Mesh(
      track(new THREE.BoxGeometry(0.94, 0.74, 0.06)),
      track(
        new THREE.MeshStandardMaterial({
          color: 0x2b2620,
          roughness: 0.5,
          metalness: 0.1,
        }),
      ),
    );
    bezel.position.set(0, 0.66, 0.46);
    computer.add(bezel);

    const screenMaterial = track(
      new THREE.MeshBasicMaterial({ color: AMBER, transparent: true, opacity: 0.9 }),
    );
    const screen = new THREE.Mesh(
      track(new THREE.PlaneGeometry(0.82, 0.62)),
      screenMaterial,
    );
    screen.position.set(0, 0.66, 0.5);
    computer.add(screen);

    // Scanlines sell the CRT without a custom shader pass.
    const scanlines = new THREE.Group();
    const scanlineMaterial = track(
      new THREE.MeshBasicMaterial({
        color: 0x3a2410,
        transparent: true,
        opacity: 0.22,
      }),
    );
    const scanlineGeometry = track(new THREE.PlaneGeometry(0.82, 0.006));
    for (let i = 0; i < 22; i += 1) {
      const line = new THREE.Mesh(scanlineGeometry, scanlineMaterial);
      line.position.set(0, 0.37 + i * 0.028, 0.501);
      scanlines.add(line);
    }
    computer.add(scanlines);

    const floppySlot = new THREE.Mesh(
      track(new THREE.BoxGeometry(0.42, 0.05, 0.04)),
      track(new THREE.MeshStandardMaterial({ color: 0x1d1a16, roughness: 0.7 })),
    );
    floppySlot.position.set(0.12, 0.18, 0.47);
    computer.add(floppySlot);

    const keyboard = new THREE.Mesh(
      track(new THREE.BoxGeometry(1.02, 0.06, 0.34)),
      caseMaterial,
    );
    keyboard.position.set(0, 0.03, 0.82);
    keyboard.rotation.x = -0.05;
    computer.add(keyboard);

    // ---------- Lamp ----------
    const lamp = new THREE.Group();
    lamp.position.set(-1.45, 0.62, 0.3);
    scene.add(lamp);

    const lampMaterial = track(
      new THREE.MeshStandardMaterial({
        color: 0x14161a,
        roughness: 0.42,
        metalness: 0.55,
      }),
    );

    const lampBase = new THREE.Mesh(
      track(new THREE.CylinderGeometry(0.19, 0.22, 0.05, 24)),
      lampMaterial,
    );
    lampBase.position.y = 0.025;
    lamp.add(lampBase);

    const lampArm = new THREE.Mesh(
      track(new THREE.CylinderGeometry(0.022, 0.022, 1.12, 12)),
      lampMaterial,
    );
    lampArm.position.set(0.1, 0.6, 0);
    lampArm.rotation.z = -0.28;
    lamp.add(lampArm);

    const lampShade = new THREE.Mesh(
      track(new THREE.ConeGeometry(0.22, 0.28, 24, 1, true)),
      track(
        new THREE.MeshStandardMaterial({
          color: 0x14161a,
          roughness: 0.4,
          metalness: 0.5,
          side: THREE.DoubleSide,
        }),
      ),
    );
    lampShade.position.set(0.3, 1.12, 0.1);
    lampShade.rotation.set(0.72, 0, -0.42);
    lamp.add(lampShade);

    const bulbMaterial = track(
      new THREE.MeshBasicMaterial({ color: 0xffd8a0, transparent: true }),
    );
    const bulb = new THREE.Mesh(
      track(new THREE.SphereGeometry(0.06, 16, 16)),
      bulbMaterial,
    );
    bulb.position.set(0.31, 1.05, 0.13);
    lamp.add(bulb);

    // ---------- Book stacks framing the desk ----------
    const bookColors = [0x8a5b45, 0x476488, 0x9c6c46, 0x527056];
    const bookGroup = new THREE.Group();
    scene.add(bookGroup);
    for (let side = 0; side < 2; side += 1) {
      const x = side === 0 ? -2.55 : 2.5;
      let y = 0.62;
      for (let i = 0; i < 5; i += 1) {
        const height = 0.085 + Math.random() * 0.045;
        const book = new THREE.Mesh(
          track(new THREE.BoxGeometry(0.62, height, 0.44)),
          track(
            new THREE.MeshStandardMaterial({
              color: bookColors[(i + side) % bookColors.length],
              roughness: 0.85,
            }),
          ),
        );
        book.position.set(x + (Math.random() - 0.5) * 0.09, y + height / 2, 0.1);
        book.rotation.y = (Math.random() - 0.5) * 0.22;
        bookGroup.add(book);
        y += height;
      }
    }

    // ---------- Stars ----------
    const starGeometry = track(new THREE.BufferGeometry());
    const starPositions = new Float32Array(900 * 3);
    for (let i = 0; i < 900; i += 1) {
      const radius = 40 + Math.random() * 45;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI * 0.42;
      starPositions[i * 3] = Math.cos(theta) * Math.sin(phi + 0.12) * radius;
      starPositions[i * 3 + 1] = Math.cos(phi) * radius * 0.75 + 2;
      starPositions[i * 3 + 2] = Math.sin(theta) * Math.sin(phi + 0.12) * radius;
    }
    starGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(starPositions, 3),
    );
    const starMaterial = track(
      new THREE.PointsMaterial({
        color: 0xe8f0ff,
        size: 0.34,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
        fog: false,
      }),
    );
    scene.add(new THREE.Points(starGeometry, starMaterial));

    // ---------- Stage 1: source panels ----------
    // Each candidate video is a small billboarded screen orbiting the machine,
    // which reads as "a source" far better than an abstract dot.
    const sourcesGroup = new THREE.Group();
    sourcesGroup.position.y = 1.44;
    scene.add(sourcesGroup);

    const frameGeometry = track(new THREE.PlaneGeometry(0.2, 0.13));
    const faceGeometry = track(new THREE.PlaneGeometry(0.17, 0.1));
    const panelMaterials: THREE.MeshBasicMaterial[] = [];
    const sourcePanels: THREE.Group[] = [];
    for (let i = 0; i < MAX_SOURCES; i += 1) {
      const frameMaterial = track(
        new THREE.MeshBasicMaterial({
          color: 0x1b1a1c,
          transparent: true,
          opacity: 0,
        }),
      );
      const faceMaterial = track(
        new THREE.MeshBasicMaterial({
          color: i % 4 === 0 ? 0xffe9c4 : AMBER_DIM,
          transparent: true,
          opacity: 0,
        }),
      );
      panelMaterials.push(frameMaterial, faceMaterial);

      const panel = new THREE.Group();
      const frame = new THREE.Mesh(frameGeometry, frameMaterial);
      const face = new THREE.Mesh(faceGeometry, faceMaterial);
      face.position.z = 0.002;
      panel.add(frame, face);

      const angle = (i / MAX_SOURCES) * Math.PI * 2;
      panel.userData.angle = angle;
      panel.userData.radius = 3 + (i % 3) * 0.32;
      panel.userData.height = 0.16 + Math.sin(i * 1.7) * 0.26;
      panel.scale.setScalar(0.01);
      sourcesGroup.add(panel);
      sourcePanels.push(panel);
    }

    // ---------- Stage 2: transcript threads ----------
    const threadMaterial = track(
      new THREE.LineBasicMaterial({
        color: 0xffca8a,
        transparent: true,
        opacity: 0,
      }),
    );
    const threadGeometry = track(new THREE.BufferGeometry());
    const threadPositions = new Float32Array(MAX_SOURCES * 2 * 3);
    threadGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(threadPositions, 3),
    );
    const threads = new THREE.LineSegments(threadGeometry, threadMaterial);
    threads.position.copy(sourcesGroup.position);
    scene.add(threads);

    // ---------- Stage 3: knowledge atoms ----------
    const atomMaterial = track(
      new THREE.MeshBasicMaterial({
        color: 0xffc27a,
        transparent: true,
        opacity: 0,
      }),
    );
    const atoms = new THREE.InstancedMesh(
      track(new THREE.SphereGeometry(0.017, 8, 8)),
      atomMaterial,
      MAX_ATOMS,
    );
    atoms.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    atoms.position.y = 1.5;
    scene.add(atoms);

    const atomSeeds = Array.from({ length: MAX_ATOMS }, (_, i) => ({
      angle: Math.random() * Math.PI * 2,
      radius: 0.7 + Math.random() * 1.15,
      height: (Math.random() - 0.5) * 0.95,
      speed: 0.12 + Math.random() * 0.3,
      cluster: i % CLUSTER_ANCHORS,
      jitterRadius: 0.05 + Math.random() * 0.14,
      jitterPhase: Math.random() * Math.PI * 2,
    }));
    const atomMatrix = new THREE.Matrix4();
    const atomPosition = new THREE.Vector3();
    const threadAnchor = new THREE.Vector3();

    // ---------- Stage 4: cluster constellation ----------
    const constellationMaterial = track(
      new THREE.LineBasicMaterial({
        color: 0x9fc4ff,
        transparent: true,
        opacity: 0,
      }),
    );
    const constellationGeometry = track(new THREE.BufferGeometry());
    const constellationPositions = new Float32Array(7 * 2 * 3);
    constellationGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(constellationPositions, 3),
    );
    const constellation = new THREE.LineSegments(
      constellationGeometry,
      constellationMaterial,
    );
    constellation.position.y = 1.5;
    scene.add(constellation);

    // ---------- Post-processing ----------
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.34, 0.55, 0.82);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());

    // Vertical FOV is fixed, so a wide-and-short window would crop into the
    // desk. Backing off with aspect keeps the machine at a comfortable size
    // and leaves room for the HUD.
    let frameDistance = 6.6;

    const resize = () => {
      const width = mount.clientWidth || window.innerWidth;
      const height = mount.clientHeight || window.innerHeight;
      renderer.setSize(width, height, false);
      composer.setSize(width, height);
      bloom.setSize(width, height);
      const aspect = width / Math.max(height, 1);
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      frameDistance = 6.6 * THREE.MathUtils.clamp(aspect / 1.7, 0.96, 1.55);
    };
    resize();
    window.addEventListener("resize", resize);

    const pointer = new THREE.Vector2();
    const onPointerMove = (event: PointerEvent) => {
      pointer.x = (event.clientX / window.innerWidth - 0.5) * 2;
      pointer.y = (event.clientY / window.innerHeight - 0.5) * 2;
    };
    if (!reduceMotion) window.addEventListener("pointermove", onPointerMove);

    // Smoothed presence per stage so visuals never pop in abruptly.
    const presence = { sources: 0, threads: 0, atoms: 0, clusters: 0, focus: 0 };
    const clock = new THREE.Clock();
    let frame = 0;

    const renderFrame = () => {
      const dt = Math.min(clock.getDelta(), 0.05);
      const time = clock.getElapsedTime();
      const { stage: liveStage, counts: liveCounts, finished: done } =
        propsRef.current;
      const stageIndex = STAGE_INDEX[liveStage] ?? 0;

      presence.sources = damp(presence.sources, stageIndex >= 0 ? 1 : 0, 1.6, dt);
      presence.threads = damp(presence.threads, stageIndex >= 1 ? 1 : 0, 1.5, dt);
      presence.atoms = damp(presence.atoms, stageIndex >= 2 ? 1 : 0, 1.2, dt);
      presence.clusters = damp(presence.clusters, stageIndex >= 3 ? 1 : 0, 1.1, dt);
      presence.focus = damp(presence.focus, stageIndex >= 4 || done ? 1 : 0, 0.9, dt);

      // Camera: dolly in from the portal, then drift with the pointer.
      const targetZ = frameDistance - presence.focus * 0.7;
      camera.position.z = damp(camera.position.z, targetZ, 1.1, dt);
      camera.position.x = damp(
        camera.position.x,
        pointer.x * 0.5 + Math.sin(time * 0.16) * 0.12,
        1.4,
        dt,
      );
      camera.position.y = damp(
        camera.position.y,
        1.78 - pointer.y * 0.18 + Math.sin(time * 0.21) * 0.05,
        1.4,
        dt,
      );
      camera.lookAt(0, 1.12, 0);

      // Screen and lamp breathe; the screen brightens as the guide is written.
      const flicker = 0.86 + Math.sin(time * 7.3) * 0.02 + Math.sin(time * 2.1) * 0.03;
      screenMaterial.opacity = flicker * (0.55 + presence.focus * 0.14);
      screenLight.intensity = 1.4 + presence.focus * 1.1 + Math.sin(time * 3.1) * 0.1;
      bulbMaterial.opacity = 0.8 + Math.sin(time * 1.7) * 0.06;
      lampLight.intensity = 20 + Math.sin(time * 1.3) * 1.4;
      bloom.strength = 0.24 + presence.focus * 0.16;

      // Sources: one panel per candidate video, gently orbiting the machine.
      const activeSources = Math.max(
        0,
        Math.min(MAX_SOURCES, liveCounts.videos ?? 0),
      );
      sourcesGroup.rotation.y = time * 0.1;
      sourcePanels.forEach((panel, i) => {
        const wanted = i < activeSources ? presence.sources : 0;
        const frameMaterial = panelMaterials[i * 2];
        const faceMaterial = panelMaterials[i * 2 + 1];
        frameMaterial.opacity = damp(frameMaterial.opacity, wanted * 0.75, 2.2, dt);
        faceMaterial.opacity = damp(faceMaterial.opacity, wanted * 0.62, 2.2, dt);
        const scale = damp(panel.scale.x, 0.35 + wanted * 0.65, 2.2, dt);
        panel.scale.setScalar(scale);
        const { angle, radius, height } = panel.userData as {
          angle: number;
          radius: number;
          height: number;
        };
        const pull = 1 - presence.focus * 0.4;
        panel.position.set(
          Math.cos(angle) * radius * pull,
          height + Math.sin(time * 0.7 + i) * 0.07,
          Math.sin(angle) * radius * pull * DEPTH_SQUASH - 0.9,
        );
        // Billboard toward the camera, cancelling the parent group's spin.
        panel.quaternion
          .copy(sourcesGroup.quaternion)
          .invert()
          .multiply(camera.quaternion);
      });

      // Threads: each transcript pulls a line from its orb into the screen.
      const activeThreads = Math.max(
        0,
        Math.min(MAX_SOURCES, liveCounts.transcripts ?? 0),
      );
      threadMaterial.opacity = damp(
        threadMaterial.opacity,
        presence.threads * 0.32,
        1.8,
        dt,
      );
      // Panel positions were just written, so refresh matrices before reading.
      sourcesGroup.updateMatrixWorld(true);
      for (let i = 0; i < MAX_SOURCES; i += 1) {
        const base = i * 6;
        const visible = i < activeThreads;
        // Thread space matches the sources group origin, so subtract it out.
        sourcePanels[i].getWorldPosition(threadAnchor);
        threadAnchor.sub(sourcesGroup.position);
        threadPositions[base] = visible ? threadAnchor.x : 0;
        threadPositions[base + 1] = visible ? threadAnchor.y : 0;
        threadPositions[base + 2] = visible ? threadAnchor.z : 0;
        // All threads terminate at the CRT, in the sources group's local space.
        threadPositions[base + 3] = 0;
        threadPositions[base + 4] = visible ? -0.23 : 0;
        threadPositions[base + 5] = visible ? 0.45 : 0;
      }
      threadGeometry.attributes.position.needsUpdate = true;

      // Atoms: extracted claims swirl, then collapse toward their clusters.
      const atomTarget = Math.max(0, Math.min(MAX_ATOMS, liveCounts.atoms ?? 0));
      // Atoms dim as they are absorbed into the guide, so the CRT stays readable.
      atomMaterial.opacity = damp(
        atomMaterial.opacity,
        presence.atoms * (0.9 - presence.focus * 0.45),
        1.4,
        dt,
      );
      atoms.count = Math.max(1, atomTarget);
      for (let i = 0; i < atoms.count; i += 1) {
        const seed = atomSeeds[i];
        const swirl = seed.angle + time * seed.speed;
        const gather = presence.clusters;

        // Loose swirl around the machine...
        const swirlX = Math.cos(swirl) * seed.radius;
        const swirlY = seed.height + Math.sin(time * 0.6 + i) * 0.03;
        const swirlZ = Math.sin(swirl) * seed.radius * DEPTH_SQUASH;

        // ...collapsing onto a camera-facing ring of cluster anchors.
        const clusterAngle = (seed.cluster / CLUSTER_ANCHORS) * Math.PI * 2;
        const drift = swirl * 0.6 + seed.jitterPhase;
        const gatherX =
          Math.cos(clusterAngle) * CLUSTER_RING_X +
          Math.cos(drift) * seed.jitterRadius;
        const gatherY =
          Math.sin(clusterAngle) * CLUSTER_RING_Y +
          Math.sin(drift * 1.3) * seed.jitterRadius;

        atomPosition.set(
          swirlX + (gatherX - swirlX) * gather,
          swirlY + (gatherY - swirlY) * gather,
          swirlZ + (CLUSTER_RING_Z - swirlZ) * gather,
        );
        atomPosition.multiplyScalar(1 - presence.focus * 0.5);
        atomMatrix.makeTranslation(
          atomPosition.x,
          atomPosition.y,
          atomPosition.z,
        );
        atoms.setMatrixAt(i, atomMatrix);
      }
      atoms.instanceMatrix.needsUpdate = true;

      // Constellation: links between cluster anchors once ideas are grouped.
      const clusterCount = Math.max(
        0,
        Math.min(CLUSTER_ANCHORS, liveCounts.clusters ?? 0),
      );
      constellationMaterial.opacity = damp(
        constellationMaterial.opacity,
        presence.clusters * 0.34,
        1.2,
        dt,
      );
      const ringScale = 1 - presence.focus * 0.5;
      for (let i = 0; i < CLUSTER_ANCHORS; i += 1) {
        const base = i * 6;
        // Only link anchors that both hold a real cluster.
        const visible = i < clusterCount && (i + 1) % CLUSTER_ANCHORS < clusterCount;
        const a = (i / CLUSTER_ANCHORS) * Math.PI * 2;
        const b = ((i + 1) / CLUSTER_ANCHORS) * Math.PI * 2;
        constellationPositions[base] = visible
          ? Math.cos(a) * CLUSTER_RING_X * ringScale
          : 0;
        constellationPositions[base + 1] = visible
          ? Math.sin(a) * CLUSTER_RING_Y * ringScale
          : 0;
        constellationPositions[base + 2] = visible ? CLUSTER_RING_Z : 0;
        constellationPositions[base + 3] = visible
          ? Math.cos(b) * CLUSTER_RING_X * ringScale
          : 0;
        constellationPositions[base + 4] = visible
          ? Math.sin(b) * CLUSTER_RING_Y * ringScale
          : 0;
        constellationPositions[base + 5] = visible ? CLUSTER_RING_Z : 0;
      }
      constellationGeometry.attributes.position.needsUpdate = true;

      composer.render();
      frame = window.requestAnimationFrame(renderFrame);
    };

    frame = window.requestAnimationFrame(renderFrame);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onPointerMove);
      disposables.forEach((item) => item.dispose());
      atoms.dispose();
      flowers.dispose();
      composer.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return <div className="research-canvas" ref={mountRef} aria-hidden="true" />;
}
