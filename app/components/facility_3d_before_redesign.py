
import streamlit.components.v1 as components


def render_facility_3d(height=760):

    html = r"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        * { box-sizing: border-box; }

        body {
          margin: 0;
          overflow: hidden;
          background: #101923;
          font-family: Arial, sans-serif;
        }

        #scene {
          width: 100%;
          height: 740px;
        }

        #legend {
          position: absolute;
          top: 15px;
          left: 15px;
          z-index: 2;
          color: #e7eef6;
          background: rgba(9, 19, 30, .92);
          border: 1px solid #34495e;
          padding: 13px 17px;
          border-radius: 8px;
          font-size: 12px;
          line-height: 2;
        }

        #legend strong {
          font-size: 12px;
          letter-spacing: 1px;
        }

        .dot {
          display: inline-block;
          width: 9px;
          height: 9px;
          border-radius: 50%;
          margin-right: 8px;
        }

        #hint {
          position: absolute;
          bottom: 14px;
          left: 15px;
          z-index: 2;
          color: #b5c4d3;
          background: rgba(9, 19, 30, .85);
          border: 1px solid #34495e;
          padding: 9px 13px;
          border-radius: 6px;
          font-size: 11px;
        }

        #title {
          position: absolute;
          top: 15px;
          right: 18px;
          color: #dce8f3;
          font-size: 13px;
          font-weight: bold;
          letter-spacing: 1px;
          z-index: 2;
        }
      </style>
    </head>

    <body>

      <div id="legend">
        <strong>PROCESS STREAMS</strong><br>
        <span class="dot" style="background:#f5a623"></span>Oil<br>
        <span class="dot" style="background:#28c8e8"></span>Gas<br>
        <span class="dot" style="background:#399cff"></span>Produced water
      </div>

      <div id="title">ONSHORE PRODUCTION FACILITY</div>

      <div id="hint">
        DRAG TO ROTATE &nbsp; | &nbsp; SCROLL TO ZOOM &nbsp; | &nbsp; RIGHT-CLICK TO PAN
      </div>

      <div id="scene"></div>

      <script type="importmap">
      {
        "imports": {
          "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
          "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
      }
      </script>

      <script type="module">

        import * as THREE from "three";
        import { OrbitControls } from "three/addons/controls/OrbitControls.js";

        const container = document.getElementById("scene");

        // ------------------------------------------------
        // SCENE AND CAMERA
        // ------------------------------------------------

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x101923);

        const camera = new THREE.PerspectiveCamera(
          42,
          container.clientWidth / container.clientHeight,
          0.1,
          1000
        );

        camera.position.set(12, 10, 15);

        const renderer = new THREE.WebGLRenderer({
          antialias: true
        });

        renderer.setPixelRatio(
          Math.min(window.devicePixelRatio, 2)
        );

        renderer.setSize(
          container.clientWidth,
          container.clientHeight
        );

        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.outputColorSpace = THREE.SRGBColorSpace;

        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(
          camera,
          renderer.domElement
        );

        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.target.set(3.5, 1.5, 0);
        controls.minDistance = 9;
        controls.maxDistance = 32;
        controls.maxPolarAngle = Math.PI * 0.48;

        // ------------------------------------------------
        // LIGHTING
        // ------------------------------------------------

        scene.add(
          new THREE.HemisphereLight(
            0xdceeff,
            0x283644,
            2.0
          )
        );

        const keyLight = new THREE.DirectionalLight(
          0xffffff,
          3
        );

        keyLight.position.set(8, 18, 12);
        keyLight.castShadow = true;
        scene.add(keyLight);

        const fillLight = new THREE.PointLight(
          0x369dff,
          45,
          40
        );

        fillLight.position.set(-5, 7, 6);
        scene.add(fillLight);

        // ------------------------------------------------
        // MATERIALS
        // ------------------------------------------------

        const steel = new THREE.MeshStandardMaterial({
          color: 0x8399aa,
          metalness: 0.82,
          roughness: 0.28
        });

        const darkSteel = new THREE.MeshStandardMaterial({
          color: 0x43596b,
          metalness: 0.8,
          roughness: 0.35
        });

        const glass = new THREE.MeshPhysicalMaterial({
          color: 0xa8d4e8,
          metalness: 0.15,
          roughness: 0.18,
          transparent: true,
          opacity: 0.3,
          transmission: 0.2,
          side: THREE.DoubleSide,
          depthWrite: false
        });

        const orange = 0xf5a623;
        const cyan = 0x28c8e8;
        const blue = 0x399cff;

        // ------------------------------------------------
        // GROUND
        // ------------------------------------------------

        const ground = new THREE.Mesh(
          new THREE.PlaneGeometry(25, 17),
          new THREE.MeshStandardMaterial({
            color: 0x172532,
            roughness: 0.95
          })
        );

        ground.rotation.x = -Math.PI / 2;
        ground.position.set(3.5, -0.2, 0);
        ground.receiveShadow = true;
        scene.add(ground);

        const grid = new THREE.GridHelper(
          25,
          25,
          0x40586b,
          0x293b49
        );

        grid.position.set(3.5, -0.17, 0);
        scene.add(grid);

        // ------------------------------------------------
        // EQUIPMENT HELPERS
        // ------------------------------------------------

        function addBox(
          x, y, z,
          width, height, depth,
          material
        ) {
          const mesh = new THREE.Mesh(
            new THREE.BoxGeometry(width, height, depth),
            material
          );

          mesh.position.set(x, y, z);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          scene.add(mesh);

          return mesh;
        }

        function addCylinder(
          x, y, z,
          radius, length,
          material,
          horizontal = false
        ) {
          const mesh = new THREE.Mesh(
            new THREE.CylinderGeometry(
              radius,
              radius,
              length,
              40
            ),
            material
          );

          mesh.position.set(x, y, z);

          if (horizontal) {
            mesh.rotation.z = Math.PI / 2;
          }

          mesh.castShadow = true;
          mesh.receiveShadow = true;
          scene.add(mesh);

          return mesh;
        }

        // Equipment labels
        function addLabel(text, x, y, z) {

          const canvas = document.createElement("canvas");
          canvas.width = 512;
          canvas.height = 100;

          const ctx = canvas.getContext("2d");

          ctx.fillStyle = "rgba(8,17,27,0.94)";
          ctx.fillRect(0, 0, 512, 100);

          ctx.strokeStyle = "#64849c";
          ctx.lineWidth = 4;
          ctx.strokeRect(2, 2, 508, 96);

          ctx.fillStyle = "#ffffff";
          ctx.font = "bold 28px Arial";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(text, 256, 50);

          const texture = new THREE.CanvasTexture(canvas);

          const sprite = new THREE.Sprite(
            new THREE.SpriteMaterial({
              map: texture,
              transparent: true,
              depthTest: false
            })
          );

          sprite.position.set(x, y, z);
          sprite.scale.set(3.0, 0.59, 1);

          scene.add(sprite);

          return sprite;
        }

        // ------------------------------------------------
        // STORAGE AND TREATMENT TANKS
        // ------------------------------------------------

        function addTank(
          x, z,
          radius, height,
          label,
          color
        ) {

          const group = new THREE.Group();

          const shell = new THREE.Mesh(
            new THREE.CylinderGeometry(
              radius,
              radius,
              height,
              48,
              1,
              false
            ),
            glass
          );

          shell.position.y = height / 2;
          group.add(shell);

          const roof = new THREE.Mesh(
            new THREE.CylinderGeometry(
              radius,
              radius,
              0.16,
              48
            ),
            new THREE.MeshStandardMaterial({
              color,
              metalness: 0.7,
              roughness: 0.3
            })
          );

          roof.position.y = height + 0.08;
          group.add(roof);

          const bottom = new THREE.Mesh(
            new THREE.CylinderGeometry(
              radius,
              radius,
              0.15,
              48
            ),
            steel
          );

          bottom.position.y = 0;
          group.add(bottom);

          // Vertical support rings
          for (let y of [0.25, height - 0.25]) {

            const ring = new THREE.Mesh(
              new THREE.TorusGeometry(
                radius,
                0.045,
                8,
                48
              ),
              darkSteel
            );

            ring.rotation.x = Math.PI / 2;
            ring.position.y = y;
            group.add(ring);
          }

          group.position.set(x, 0, z);
          scene.add(group);

          addLabel(
            label,
            x,
            height + 0.8,
            z
          );

          return group;
        }

        // ------------------------------------------------
        // PRODUCTION WELL
        // ------------------------------------------------

        addCylinder(
          -5, 1.7, 0,
          0.38, 3.4,
          steel
        );

        addCylinder(
          -5, 1.7, 0,
          0.26, 3.45,
          glass
        );

        addBox(
          -5, 3.5, 0,
          0.9, 0.35, 0.9,
          darkSteel
        );

        addLabel(
          "PRODUCTION WELL",
          -5, 4.1, 0
        );

        // ------------------------------------------------
        // INLET MANIFOLD
        // ------------------------------------------------

        addCylinder(
          -2.6, 1.7, 0,
          0.25, 3.3,
          steel,
          true
        );

        addCylinder(
          -2.6, 1.7, 0,
          0.15, 3.35,
          glass,
          true
        );

        addLabel(
          "INLET MANIFOLD",
          -2.6, 2.7, 0
        );

        // ------------------------------------------------
        // THREE-PHASE SEPARATOR
        // ------------------------------------------------

        const separator = new THREE.Group();

        const separatorBody = new THREE.Mesh(
          new THREE.CylinderGeometry(
            0.9, 0.9, 4.5, 48
          ),
          glass
        );

        separatorBody.rotation.z = Math.PI / 2;
        separatorBody.position.y = 2.0;
        separator.add(separatorBody);

        for (const x of [-2.25, 2.25]) {

          const cap = new THREE.Mesh(
            new THREE.SphereGeometry(
              0.9, 32, 24
            ),
            glass
          );

          cap.position.set(x, 2.0, 0);
          cap.scale.set(0.27, 1, 1);

          separator.add(cap);
        }

        // Separator supports
        for (const x of [-1.5, 1.5]) {

          const support = new THREE.Mesh(
            new THREE.BoxGeometry(
              0.2, 1.1, 0.2
            ),
            darkSteel
          );

          support.position.set(x, 0.55, 0);
          separator.add(support);
        }

        // Separator top instrumentation
        const nozzle = new THREE.Mesh(
          new THREE.CylinderGeometry(
            0.14, 0.14, 0.7, 20
          ),
          steel
        );

        nozzle.position.set(0, 2.85, 0);
        separator.add(nozzle);

        separator.position.set(1.4, 0, 0);
        scene.add(separator);

        addLabel(
          "THREE-PHASE SEPARATOR",
          1.4, 3.8, 0
        );

        // ------------------------------------------------
        // OIL PROCESSING
        // ------------------------------------------------

        addTank(
          5.5, -4,
          0.9, 3.1,
          "OIL TREATMENT",
          0xe5a23b
        );

        addTank(
          9.5, -4,
          1.3, 4.0,
          "OIL STORAGE",
          0xd8a441
        );

        // ------------------------------------------------
        // GAS COMPRESSOR
        // ------------------------------------------------

        const compressor = new THREE.Group();

        const base = new THREE.Mesh(
          new THREE.BoxGeometry(
            2.5, 0.35, 1.8
          ),
          darkSteel
        );

        base.position.y = 0.2;
        compressor.add(base);

        const housing = new THREE.Mesh(
          new THREE.BoxGeometry(
            2.0, 1.3, 1.4
          ),
          steel
        );

        housing.position.y = 1.0;
        compressor.add(housing);

        const motor = new THREE.Mesh(
          new THREE.CylinderGeometry(
            0.5, 0.5, 1.25, 36
          ),
          new THREE.MeshStandardMaterial({
            color: 0x9cb1c1,
            metalness: 0.8,
            roughness: 0.3
          })
        );

        motor.rotation.z = Math.PI / 2;
        motor.position.set(0, 1.9, 0);
        compressor.add(motor);

        compressor.position.set(5.3, 0, 2.8);
        scene.add(compressor);

        addLabel(
          "GAS COMPRESSOR",
          5.3, 3.0, 2.8
        );

        // ------------------------------------------------
        // GAS TREATMENT
        // ------------------------------------------------

        addTank(
          9.0, 2.8,
          0.9, 3.6,
          "GAS TREATMENT",
          0x38b8ca
        );

        // ------------------------------------------------
        // PRODUCED WATER TREATMENT
        // ------------------------------------------------

        addTank(
          5.5, 5.7,
          1.05, 2.8,
          "WATER TREATMENT",
          0x398fdb
        );

        // ------------------------------------------------
        // PIPELINE SYSTEM
        // ------------------------------------------------

        const flowSystems = [];

        function addPipeline(
          points,
          color,
          radius = 0.13,
          particleCount = 10
        ) {

          const curve = new THREE.CatmullRomCurve3(
            points.map(
              p => new THREE.Vector3(
                p[0], p[1], p[2]
              )
            )
          );

          // Outer transparent pipe
          const outer = new THREE.Mesh(
            new THREE.TubeGeometry(
              curve, 70, radius, 14, false
            ),
            new THREE.MeshPhysicalMaterial({
              color: 0xaac1d3,
              metalness: 0.6,
              roughness: 0.2,
              transparent: true,
              opacity: 0.42
            })
          );

          scene.add(outer);

          // Internal stream
          const inner = new THREE.Mesh(
            new THREE.TubeGeometry(
              curve, 70, radius * 0.43, 12, false
            ),
            new THREE.MeshStandardMaterial({
              color,
              emissive: color,
              emissiveIntensity: 0.4,
              metalness: 0.15,
              roughness: 0.3
            })
          );

          scene.add(inner);

          // Flow particles
          const particles = [];

          for (let i = 0; i < particleCount; i++) {

            const particle = new THREE.Mesh(
              new THREE.SphereGeometry(
                radius * 0.8, 12, 12
              ),
              new THREE.MeshBasicMaterial({
                color
              })
            );

            scene.add(particle);

            particles.push({
              mesh: particle,
              offset: i / particleCount
            });
          }

          flowSystems.push({
            curve,
            particles
          });
        }

        // Production inlet
        addPipeline([
          [-5, 1.7, 0],
          [-3.8, 1.7, 0],
          [-2.6, 1.7, 0],
          [-1, 1.8, 0],
          [1.4, 2.0, 0]
        ], orange, 0.17, 14);

        // Oil processing branch
        addPipeline([
          [1.4, 1.8, 0],
          [3.2, 1.6, -1.2],
          [5.5, 1.5, -4]
        ], orange);

        // Oil storage connection
        addPipeline([
          [5.5, 1.5, -4],
          [7.4, 1.5, -4],
          [9.5, 1.5, -4]
        ], orange);

        // Gas compression branch
        addPipeline([
          [1.4, 2.4, 0],
          [3.0, 2.6, 1.4],
          [5.3, 1.5, 2.8]
        ], cyan);

        // Gas treatment and export
        addPipeline([
          [5.3, 1.5, 2.8],
          [7.0, 1.7, 2.8],
          [9.0, 1.8, 2.8],
          [12.0, 1.8, 2.8]
        ], cyan);

        // Produced water branch
        addPipeline([
          [1.4, 1.1, 0],
          [3.2, 0.9, 2.5],
          [5.5, 1.4, 5.7]
        ], blue);

        // Export labels
        addLabel(
          "OIL EXPORT",
          12, 2.8, -4
        );

        addLabel(
          "GAS EXPORT",
          12, 2.8, 2.8
        );

        // ------------------------------------------------
        // ANIMATION
        // ------------------------------------------------

        function animate() {

          requestAnimationFrame(animate);

          const time = performance.now() * 0.00012;

          flowSystems.forEach(system => {

            system.particles.forEach(item => {

              const t = (
                time + item.offset
              ) % 1;

              item.mesh.position.copy(
                system.curve.getPointAt(t)
              );

            });

          });

          controls.update();

          renderer.render(
            scene,
            camera
          );
        }

        animate();

        // ------------------------------------------------
        // RESPONSIVE RESIZING
        // ------------------------------------------------

        window.addEventListener(
          "resize",
          () => {

            const width = container.clientWidth;
            const height = container.clientHeight;

            camera.aspect = width / height;
            camera.updateProjectionMatrix();

            renderer.setSize(
              width,
              height
            );

          }
        );

      </script>
    </body>
    </html>
    """

    components.html(
        html,
        height=height,
        scrolling=False
    )
