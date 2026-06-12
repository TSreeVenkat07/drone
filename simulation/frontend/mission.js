const API_URL = `/api/start_mission`;
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL_BASE = `${protocol}//${window.location.host}/ws/mission/`;

// State
let currentScenario = null;
let victims = []; // [{x, y}]
let missionId = null;
let socket = null;

const GRID_SIZE = 32;
const CELL_SIZE = 20; // Used for placement canvas

// Colors
const COLORS = {
    collapse: 0x4a4a4a,
    wildfire: 0x8B3A0F,
    flood: 0x1a3a5c,
    coverageOverlay: 0xffffff,
    victimUnrescued: 0xffffff,
    victimRescued: 0x00FF88,
    drone: 0xFFD700,
    gridLines: 0x000000
};

// DOM Elements
const screen1 = document.getElementById('screen1');
const screen2 = document.getElementById('screen2');
const screen3 = document.getElementById('screen3');

const placementCanvas = document.getElementById('placementCanvas');
const pCtx = placementCanvas.getContext('2d');

const sim3d = document.getElementById('sim3d');
const launchBtn = document.getElementById('launchBtn');

// Three.js variables
let scene, camera, renderer, controls;
let droneMeshes = [];
let victimMeshes = [];
let coverageMeshes = new Map();
let basePlane;

// Navigation
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
}

// SCREEN 1 -> SCREEN 2
function selectMap(scenario) {
    currentScenario = scenario;
    victims = [];
    showScreen('screen2');
    drawPlacementGrid();
}

// SCREEN 2: Placement
function drawPlacementGrid() {
    pCtx.clearRect(0, 0, placementCanvas.width, placementCanvas.height);
    const baseColorStr = '#' + COLORS[currentScenario].toString(16).padStart(6, '0');
    
    // Fill base
    pCtx.fillStyle = baseColorStr;
    pCtx.fillRect(0, 0, placementCanvas.width, placementCanvas.height);
    
    // Draw lines
    pCtx.strokeStyle = 'rgba(0,0,0,0.2)';
    for (let i = 0; i <= GRID_SIZE; i++) {
        pCtx.beginPath();
        pCtx.moveTo(i * CELL_SIZE, 0);
        pCtx.lineTo(i * CELL_SIZE, placementCanvas.height);
        pCtx.stroke();
        
        pCtx.beginPath();
        pCtx.moveTo(0, i * CELL_SIZE);
        pCtx.lineTo(placementCanvas.width, i * CELL_SIZE);
        pCtx.stroke();
    }
    
    // Draw victims
    victims.forEach(v => {
        pCtx.fillStyle = '#FFFFFF';
        pCtx.beginPath();
        pCtx.arc(v.x * CELL_SIZE + CELL_SIZE/2, v.y * CELL_SIZE + CELL_SIZE/2, CELL_SIZE/3, 0, Math.PI*2);
        pCtx.fill();
        
        pCtx.fillStyle = 'red';
        pCtx.font = 'bold 12px Arial';
        pCtx.textAlign = 'center';
        pCtx.textBaseline = 'middle';
        pCtx.fillText('!', v.x * CELL_SIZE + CELL_SIZE/2, v.y * CELL_SIZE + CELL_SIZE/2);
    });
}

placementCanvas.addEventListener('click', (e) => {
    const rect = placementCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / CELL_SIZE);
    const y = Math.floor((e.clientY - rect.top) / CELL_SIZE);
    
    if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
        const existingIdx = victims.findIndex(v => v.x === x && v.y === y);
        if (existingIdx >= 0) {
            victims.splice(existingIdx, 1);
        } else {
            victims.push({x, y});
        }
        
        launchBtn.style.display = victims.length > 0 ? 'block' : 'none';
        drawPlacementGrid();
    }
});

// Setup 3D Scene
function init3D() {
    if (scene) return; // already initialized
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0b10);
    
    const width = sim3d.clientWidth;
    const height = sim3d.clientHeight;
    
    camera = new THREE.PerspectiveCamera(45, width / height, 1, 1000);
    // Position camera diagonally looking down at the center of the 32x32 grid
    camera.position.set(GRID_SIZE/2, GRID_SIZE, GRID_SIZE + 10);
    
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    sim3d.appendChild(renderer.domElement);
    
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(GRID_SIZE/2, 0, GRID_SIZE/2);
    controls.update();
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    scene.add(dirLight);
    
    // Base Plane
    const planeGeo = new THREE.PlaneGeometry(GRID_SIZE, GRID_SIZE);
    const planeMat = new THREE.MeshStandardMaterial({ color: COLORS[currentScenario] });
    basePlane = new THREE.Mesh(planeGeo, planeMat);
    basePlane.rotation.x = -Math.PI / 2;
    // Align plane so (0,0) is at top-left conceptually, but in 3D center is usually 0,0.
    // Let's position it so that top-left is 0,0 and bottom-right is 32,32
    basePlane.position.set(GRID_SIZE/2, -0.1, GRID_SIZE/2);
    scene.add(basePlane);
    
    // Grid Helper
    const gridHelper = new THREE.GridHelper(GRID_SIZE, GRID_SIZE, COLORS.gridLines, COLORS.gridLines);
    gridHelper.position.set(GRID_SIZE/2, 0, GRID_SIZE/2);
    scene.add(gridHelper);
    
    // Create initial victims
    const sphereGeo = new THREE.SphereGeometry(0.4, 16, 16);
    victims.forEach((v, idx) => {
        const mat = new THREE.MeshStandardMaterial({ color: COLORS.victimUnrescued });
        const mesh = new THREE.Mesh(sphereGeo, mat);
        // +0.5 to center in cell
        mesh.position.set(v.x + 0.5, 0.4, v.y + 0.5);
        scene.add(mesh);
        victimMeshes.push({ idx, mesh, data: v });
    });
    
    // Create 4 drones
    const droneGeo = new THREE.ConeGeometry(0.4, 1, 8);
    // Rotate cone to point forward (optional, but pointing down or forward)
    droneGeo.rotateX(Math.PI / 2);
    for(let i=0; i<4; i++) {
        const mat = new THREE.MeshStandardMaterial({ color: COLORS.drone });
        const mesh = new THREE.Mesh(droneGeo, mat);
        mesh.position.set(GRID_SIZE/2, 2, GRID_SIZE/2); // start center high
        scene.add(mesh);
        droneMeshes.push(mesh);
    }
    
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// SCREEN 2 -> SCREEN 3
async function launchMission() {
    try {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ scenario: currentScenario, victims })
        });
        const data = await res.json();
        missionId = data.mission_id;
        
        showScreen('screen3');
        
        // Show loading temporarily inside sim3d
        sim3d.innerHTML = `<div style="display:flex; flex-direction:column; height:100%; align-items:center; justify-content:center; color:white;">
            <h2 style="margin-bottom:10px;">Loading 3D Environment...</h2>
            <p style="color:#9ca3af;">(This may take 5-15 seconds on CPU)</p>
        </div>`;
        
        startWebSocket();
    } catch (err) {
        console.error("Failed to launch mission:", err);
        alert("Failed to connect to backend server.");
    }
}

// SCREEN 3: Live Simulation
function startWebSocket() {
    window.loggedVictims = new Set();
    lastFoundCount = 0;
    obstaclesDrawn = false;
    document.getElementById('gpsLogs').innerHTML = '';
    
    socket = new WebSocket(WS_URL_BASE + missionId);
    
    socket.onmessage = (event) => {
        const state = JSON.parse(event.data);
        
        if (!scene) {
            sim3d.innerHTML = ''; // clear loading
            init3D();
        }
        
        renderSimGrid(state);
        updateStats(state);
    };
    
    socket.onclose = () => {
        console.log("WebSocket closed");
        alert("Mission ended or connection lost.");
    };
}

let lastFoundCount = 0;
let obstaclesDrawn = false;

function updateStats(state) {
    document.getElementById('hudCoverage').innerHTML = `
        <div style="font-size: 0.85em; color: #aaa;">Total Map Blocks: ${state.stats.total}</div>
        <div style="font-size: 0.85em; color: #aaa;">Obstacle Blocks: ${state.stats.obstacles}</div>
        <div style="font-size: 0.85em; color: #aaa;">Free Space Blocks: ${state.stats.navigable}</div>
        <div style="margin-top: 4px; font-weight: bold;">Explored: ${state.stats.covered} / ${state.stats.navigable}</div>
    `;
    document.getElementById('hudVictims').textContent = `${state.stats.found} / ${victims.length}`;
    document.getElementById('hudSteps').textContent = `${state.stats.steps}`;
    
    const fleetList = document.getElementById('fleetList');
    fleetList.innerHTML = state.drones.map(d => `
        <div class="drone-item">
            <span>Drone ${d.id} [${d.x}, ${d.y}]</span>
            <span>${d.scenario_known ? '✅ ' + d.scenario : '❓'}</span>
        </div>
    `).join('');
    
    // Robustly check for found victims and update colors/logs
    if (!window.loggedVictims) window.loggedVictims = new Set();
    
    state.victims.forEach((v, idx) => {
        // Sync 3D Sphere Color
        if (v.found) {
            victimMeshes[idx].mesh.material.color.setHex(COLORS.victimRescued);
            
            // Only log GPS once per victim
            if (!window.loggedVictims.has(idx)) {
                window.loggedVictims.add(idx);
                
                const lat = 17.3850 + (v.y * 0.0001);
                const lon = 78.4867 + (v.x * 0.0001);
                
                const logBox = document.getElementById('gpsLogs');
                const p = document.createElement('div');
                p.style.color = '#' + COLORS.victimRescued.toString(16).padStart(6, '0');
                p.style.marginBottom = '5px';
                p.innerHTML = `[Tick ${state.tick}] Victim found - Grid [${v.x}, ${v.y}]<br>&nbsp;&nbsp;&nbsp;GPS: [${lat.toFixed(6)}, ${lon.toFixed(6)}]`;
                logBox.appendChild(p);
                logBox.scrollTop = logBox.scrollHeight;
            }
        } else {
            victimMeshes[idx].mesh.material.color.setHex(COLORS.victimUnrescued);
        }
    });
    
    lastFoundCount = state.stats.found;
}

function renderSimGrid(state) {
    if (!window.visitedCells) window.visitedCells = new Set();
    
    // Draw obstacles on first tick if provided
    if (state.obstacles && !obstaclesDrawn && scene) {
        obstaclesDrawn = true;
        const obsMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 });
        const obsGeo = new THREE.BoxGeometry(1, 1, 1);
        for (let r = 0; r < GRID_SIZE; r++) {
            for (let c = 0; c < GRID_SIZE; c++) {
                if (state.obstacles[r][c] >= 1.0) {
                    const mesh = new THREE.Mesh(obsGeo, obsMat);
                    mesh.position.set(c + 0.5, 0.5, r + 0.5);
                    scene.add(mesh);
                }
            }
        }
    }
    
    // Update drones
    state.drones.forEach((d, i) => {
        const mesh = droneMeshes[i];
        if (mesh) {
            // Target position based on grid
            const targetX = d.x + 0.5;
            const targetZ = d.y + 0.5;
            
            // Look at direction
            mesh.lookAt(targetX, mesh.position.y, targetZ);
            
            // Move there
            mesh.position.set(targetX, 1.0, targetZ);
            
            // Add coverage trail
            const key = `${d.x},${d.y}`;
            if (!window.visitedCells.has(key)) {
                window.visitedCells.add(key);
                
                // Create a small translucent plane or cube for coverage
                const covGeo = new THREE.PlaneGeometry(1, 1);
                const covMat = new THREE.MeshBasicMaterial({ color: COLORS.coverageOverlay, transparent: true, opacity: 0.3 });
                const covMesh = new THREE.Mesh(covGeo, covMat);
                covMesh.rotation.x = -Math.PI / 2;
                covMesh.position.set(targetX, 0.01, targetZ);
                scene.add(covMesh);
                coverageMeshes.set(key, covMesh);
            }
        }
    });
}
