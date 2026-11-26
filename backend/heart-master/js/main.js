import * as THREE from '../build/three.module.js';

import { DDSLoader } from './DDSLoader.js';
import { MTLLoader } from './MTLLoader.js';
import { OBJLoader } from './OBJLoader.js';
import { TrackballControls } from './TrackballControls.js';
	
const BASE_URL = 'http://mattschroyer.github.io/heart/models/';

// Read initial configuration (disease state, progression, lifestyle toggles, etc.)
// that may be passed via query parameters from the hosting app (Streamlit).
const initialConfig = (typeof window !== 'undefined' && window.HEART_VIEWER_CONFIG) || {};

const appContainer = document.getElementById("heart-app");
const annotationPanel = (() => {
	const container = document.getElementById('annotation_panel');
	if (!container) {
		// Create lightweight no-op stubs so the 3D viewer can run even when
		// the annotation UI has been stripped from index.html.
		const fakeClassList = { add: () => {}, remove: () => {} };
		const fakeContainer = { classList: fakeClassList };
		const fakeTextEl = { textContent: '' };
		const fakeButton = { addEventListener: () => {} };
		return {
			container: fakeContainer,
			title: fakeTextEl,
			description: fakeTextEl,
			disease: fakeTextEl,
			closeBtn: fakeButton,
		};
	}
	return {
		container,
		title: document.getElementById('annotation_title'),
		description: document.getElementById('annotation_description'),
		disease: document.getElementById('annotation_disease'),
		closeBtn: document.getElementById('annotation_close'),
	};
})();
const progressionSlider = document.getElementById('progression_slider');
const smokingToggle = document.getElementById('factor_smoking');
const bpToggle = document.getElementById('factor_bp');
const cholesterolToggle = document.getElementById('factor_cholesterol');
const alcoholToggle = document.getElementById('factor_alcohol');
const exerciseToggle = document.getElementById('factor_exercise');
const bloodFlowToggle = document.getElementById('toggle_blood_flow');
const realismToggle = document.getElementById('toggle_realism');
const rhythmToggle = document.getElementById('toggle_rhythm');
const riskIndicatorValue = document.getElementById('risk_indicator_value');
const riskIndicatorDetail = document.getElementById('risk_indicator_detail');
const guidedTourBtn = document.getElementById('guided_tour_btn');

// Health factor indicator elements
let smokingIndicator = null;
let cholesterolIndicator = null;
let bpIndicator = null;
let alcoholIndicator = null;

let camera, scene, renderer;
let controls;
let raycaster;
let mouse;
let heartPart;
let heartIsOpen = false;
let openHeartRequested = initialConfig.model === 'openheart';
let currentDiseaseState = initialConfig.state || 'healthy';
const heartMeshes = [];
const hotspotScreenPositions = [];
let activeHotspotIndex = null;
let progressionValue = Number.isFinite(initialConfig.progress) ? initialConfig.progress : 0;
const lifestyleFactors = {
	smoking: false,
	bp: false,
	cholesterol: false,
	alcohol: false,
	exercise: true
};
lifestyleFactors.smoking = !!initialConfig.smoking;
lifestyleFactors.bp = !!initialConfig.bp;
lifestyleFactors.cholesterol = !!initialConfig.cholesterol;
lifestyleFactors.alcohol = !!initialConfig.alcohol;
lifestyleFactors.exercise = initialConfig.exercise !== undefined ? !!initialConfig.exercise : true;
let guidedTourActive = false;
let flowSpeed = 0.06;
let bloodFlowEnabled = !!initialConfig.bloodFlow;
let realismEnabled = !!initialConfig.realism;
let rhythmEnabled = !!initialConfig.rhythm; // Initialize from URL parameters
let latestRiskScore = 0;
const diseaseOverlays = {};
let activeOverlayGroup = null;
const lifestyleOverlays = {};
const bloodFlowTracks = [];
let flowRoot = null;
const BLOOD_CELL_COUNT = 120;
let focusArrow = null;
let currentFocusTarget = null;
let focusPointT = null;

const plaquePositions = [
	new THREE.Vector3(-4.5, 5.4, -0.5),
	new THREE.Vector3(-2.6, 7.1, 2.2),
	new THREE.Vector3(-1.8, 3.9, 3.6)
];
const valvePosition = new THREE.Vector3(-3.6, 1.0, 3.2);
const infarctPosition = new THREE.Vector3(1.8, 2.4, 5.8);

const lifestyleFocusTargets = {
	smoking: new THREE.Vector3(-4.0, 5.0, 0.3),
	bp: valvePosition.clone(),
	cholesterol: new THREE.Vector3(-2.0, 6.5, 3.0),
	alcohol: infarctPosition.clone(),
	exercise: new THREE.Vector3(0.0, -1.5, 0.0)
};

const annotationContent = [
	{
		title: 'Aorta',
		description: 'Main artery carrying oxygen-rich blood from the left ventricle to the body.',
		disease: 'Plaque build-up narrows this vessel in atherosclerosis, restricting blood flow to organs.'
	},
	{
		title: 'Arch of the Aorta',
		description: 'Curved segment supplying the head, neck, and arms.',
		disease: 'Hypertension and plaque can stiffen the arch, increasing cardiac workload.'
	},
	{
		title: 'Left Auricular Appendage',
		description: 'Muscular pouch on the left atrium aiding blood collection.',
		disease: 'Clots may form here during atrial fibrillation, risking stroke.'
	},
	{
		title: 'Pulmonary Artery',
		description: 'Carries deoxygenated blood from the heart to the lungs.',
		disease: 'Pulmonary hypertension thickens artery walls, straining the right ventricle.'
	},
	{
		title: 'Pulmonary Veins',
		description: 'Return oxygenated blood from the lungs to the heart.',
		disease: 'Fibrosis or stenosis disrupts filling of the left atrium and lowers output.'
	},
	{
		title: 'Superior Vena Cava',
		description: 'Brings blood from the upper body into the right atrium.',
		disease: 'Blockage or compression causes swelling and reduced venous return.'
	}
];

// Create health factor indicator elements
function createHealthFactorIndicators() {
    const existingIndicators = document.querySelectorAll('.health-factor-indicator');
    existingIndicators.forEach((indicator) => indicator.remove());

    const appContainer = document.body;

    if (!lifestyleFactors.exercise) {
        exerciseIndicator = document.createElement('div');
        exerciseIndicator.className = 'health-factor-indicator exercise-indicator';
        exerciseIndicator.textContent = 'Sedentary: Reduced stamina';
        appContainer.appendChild(exerciseIndicator);
    }

    const severeFactors = [lifestyleFactors.smoking, lifestyleFactors.bp, lifestyleFactors.cholesterol, lifestyleFactors.alcohol].filter(Boolean).length;
    if (severeFactors >= 3) {
        const bloodFlowIndicator = document.createElement('div');
        bloodFlowIndicator.className = 'health-factor-indicator bloodflow-warning';
        bloodFlowIndicator.textContent = severeFactors === 4 ? 'Severe risk: Blood flow critically slowed' : 'High risk: Blood flow noticeably reduced';
        appContainer.appendChild(bloodFlowIndicator);
    }
    
    if (lifestyleFactors.smoking) {
        smokingIndicator = document.createElement('div');
        smokingIndicator.className = 'health-factor-indicator smoking-indicator';
        smokingIndicator.textContent = 'Smoking: Arterial damage';
        appContainer.appendChild(smokingIndicator);
    }
    
    if (lifestyleFactors.cholesterol) {
        cholesterolIndicator = document.createElement('div');
        cholesterolIndicator.className = 'health-factor-indicator cholesterol-indicator';
        cholesterolIndicator.textContent = 'High Cholesterol: Plaque buildup';
        appContainer.appendChild(cholesterolIndicator);
    }
    
    if (lifestyleFactors.bp) {
        bpIndicator = document.createElement('div');
        bpIndicator.className = 'health-factor-indicator bp-indicator';
        bpIndicator.textContent = 'High BP: Increased strain';
        appContainer.appendChild(bpIndicator);
    }
    
    if (lifestyleFactors.alcohol) {
        alcoholIndicator = document.createElement('div');
        alcoholIndicator.className = 'health-factor-indicator alcohol-indicator';
        alcoholIndicator.textContent = 'Alcohol: Irregular rhythm';
        appContainer.appendChild(alcoholIndicator);
    }
    
    // Add rhythm indicator when any factor is active
    let rhythmIndicator = document.getElementById('rhythm-indicator');
    if (rhythmIndicator) {
        rhythmIndicator.remove();
    }
    
    // Show rhythm indicator when rhythm is enabled
    if (rhythmEnabled) {
        rhythmIndicator = document.createElement('div');
        rhythmIndicator.id = 'rhythm-indicator';
        rhythmIndicator.className = 'health-factor-indicator rhythm-indicator';
        
        // Set rhythm indicator text based on active factors or default to healthy rhythm
        if (currentDiseaseState !== 'healthy') {
            rhythmIndicator.textContent = `Disease: ${diseaseStates[currentDiseaseState].label} rhythm`;
        } else if (lifestyleFactors.alcohol) {
            rhythmIndicator.textContent = 'Alcohol: Irregular rhythm';
        } else if (lifestyleFactors.bp) {
            rhythmIndicator.textContent = 'High BP: Rapid rhythm';
        } else if (lifestyleFactors.smoking) {
            rhythmIndicator.textContent = 'Smoking: Stress rhythm';
        } else if (lifestyleFactors.cholesterol) {
            rhythmIndicator.textContent = 'Cholesterol: Constricted rhythm';
        } else if (!lifestyleFactors.exercise) {
            rhythmIndicator.textContent = 'Sedentary: Slow rhythm';
        } else {
            rhythmIndicator.textContent = 'Active: Healthy rhythm';
        }
        
        appContainer.appendChild(rhythmIndicator);
    }
    
    // Hide indicators if no factors are enabled
    if (!lifestyleFactors.smoking && !lifestyleFactors.cholesterol && 
        !lifestyleFactors.bp && !lifestyleFactors.alcohol && 
        currentDiseaseState === 'healthy') {
        // All indicators are already removed above
    }
}

function createBloodFlowIndicator() {
    const p = allMyPins;
    const points = [
        p[5].clone(),
        new THREE.Vector3(-2.2, 3.0, -2.0),
        p[3].clone(),
        valvePosition.clone(),
        p[4].clone(),
        new THREE.Vector3(-0.6, 2.6, 2.8),
        p[2].clone(),
        infarctPosition.clone(),
        p[1].clone(),
        new THREE.Vector3(0.6, 3.6, 2.0),
        p[0].clone()
    ];
    createBloodFlowTrack(points);
}

function updateBloodFlowVisual(riskScore) {
    if (!bloodFlowEnabled) {
        return;
    }
    const inBase = new THREE.Color(0x8b0000);
    const outBase = new THREE.Color(0xff4c4c);
    const riskTint = new THREE.Color(0x961313);
    flowSpeed = THREE.MathUtils.lerp(0.045, 0.006, riskScore);

    const severeFactors = [
        lifestyleFactors.smoking,
        lifestyleFactors.bp,
        lifestyleFactors.cholesterol,
        lifestyleFactors.alcohol,
    ].filter(Boolean).length;

    if (severeFactors >= 4) {
        flowSpeed *= 0.25;
    } else if (severeFactors === 3) {
        flowSpeed *= 0.4;
    } else if (severeFactors === 2) {
        flowSpeed *= 0.65;
    }

    bloodFlowTracks.forEach((track) => {
        const inColor = inBase.clone().lerp(riskTint, riskScore * 0.5);
        const outColor = outBase.clone().lerp(riskTint, riskScore * 0.5);
        track.materialIn.color.copy(inColor);
        track.materialOut.color.copy(outColor);
        [track.materialIn, track.materialOut].forEach((mat) => {
            mat.dashSize = THREE.MathUtils.lerp(2.4, 1.0, riskScore);
            mat.gapSize = THREE.MathUtils.lerp(0.6, 1.6, riskScore);
            mat.opacity = THREE.MathUtils.lerp(0.95, 0.5, riskScore);
            mat.needsUpdate = true;
        });
        // keep line scale constant under camera zoom
        if (currentDiseaseState === 'myocardial_infarction') {
            track.materialOut.opacity = 0.35;
        }
    });
}

function createBloodFlowTrack(points) {
    const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.4);

    const pvPoint = points[4].clone();
    let arterialStartT = 0.5;
    {
        const samples = 200;
        let minD = Infinity;
        let minU = 0;
        for (let i = 0; i <= samples; i++) {
            const u = i / samples;
            const sPoint = curve.getPointAt(u);
            const d = sPoint.distanceTo(pvPoint);
            if (d < minD) {
                minD = d;
                minU = u;
            }
        }
        arterialStartT = minU;
    }

    const makeLine = (pts, baseColor) => {
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineDashedMaterial({
            color: baseColor,
            dashSize: 2,
            gapSize: 1,
            linewidth: 2,
            transparent: true,
            opacity: 0.9,
            depthTest: false
        });
        const ln = new THREE.Line(geom, mat);
        ln.computeLineDistances();
        ln.renderOrder = 10;
        (flowRoot || scene).add(ln);
        return { geom, mat, ln };
    };

    const sampleCount = 160;
    const inPoints = [];
    const outPoints = [];
    for (let i = 0; i <= sampleCount; i++) {
        const u = i / sampleCount;
        const p = curve.getPointAt(u);
        if (u <= arterialStartT) inPoints.push(p);
        else outPoints.push(p);
    }

    const darkInColor = 0x8b0000;
    const brightOutColor = 0xff4c4c;
    const { mat: materialIn, ln: lineIn } = makeLine(inPoints, darkInColor);
    const { mat: materialOut, ln: lineOut } = makeLine(outPoints, brightOutColor);

    const cellGeometry = new THREE.CylinderGeometry(0.18, 0.18, 0.1, 12);
    cellGeometry.rotateZ(Math.PI / 2);
    const cellMaterial = new THREE.MeshStandardMaterial({ color: 0xff4c4c, emissive: 0x5a0000, metalness: 0.15, roughness: 0.4, depthTest: false, transparent: true, opacity: 0.95 });
    const cells = [];
    for (let i = 0; i < BLOOD_CELL_COUNT; i++) {
        const mesh = new THREE.Mesh(cellGeometry, cellMaterial.clone());
        mesh.castShadow = false;
        mesh.renderOrder = 11;
        (flowRoot || scene).add(mesh);
        cells.push({ mesh, offset: i / BLOOD_CELL_COUNT });
    }

    let occlusionT = 0.6;
    {
        const samples = 200;
        let minD = Infinity;
        let minU = 0;
        for (let i = 0; i <= samples; i++) {
            const u = i / samples;
            const sPoint = curve.getPointAt(u);
            const d = sPoint.distanceTo(infarctPosition);
            if (d < minD) {
                minD = d;
                minU = u;
            }
        }
        occlusionT = minU;
    }
    bloodFlowTracks.push({ curve, materialIn, lineIn, materialOut, lineOut, cells, arterialStartT, occlusionT });
}

function updateBloodCellFlow() {
    if (!bloodFlowEnabled || !bloodFlowTracks.length) {
        return;
    }
    const time = performance.now() * 0.00012;
    let travelSpeed = THREE.MathUtils.lerp(0.18, 0.04, latestRiskScore);
    if (lifestyleFactors.bp) {
        travelSpeed *= 1.25;
    }
    if (!lifestyleFactors.exercise) {
        travelSpeed *= 0.85;
    }
    const inBase = new THREE.Color(0x8b0000);
    const outBase = new THREE.Color(0xff4c4c);
    const riskTint = new THREE.Color(0x961313);
    bloodFlowTracks.forEach((track) => {
        const localWindow = realismEnabled && focusPointT != null ? 0.06 : null;
        track.cells.forEach((cell) => {
            const t = (cell.offset + time * travelSpeed) % 1;
            let point = track.curve.getPointAt(t);
            if (realismEnabled && currentDiseaseState === 'myocardial_infarction' && t > track.occlusionT) {
                point = track.curve.getPointAt(track.occlusionT);
            }
            if (localWindow != null) {
                const dt = Math.abs(t - focusPointT);
                const dtWrap = Math.min(dt, 1 - dt);
                const visibleLocal = dtWrap <= localWindow;
                cell.mesh.visible = visibleLocal && bloodFlowEnabled;
                if (!visibleLocal) {
                    return;
                }
            } else {
                cell.mesh.visible = bloodFlowEnabled;
            }
            cell.mesh.position.copy(point);
            const tangent = track.curve.getTangentAt(t).normalize();
            const quaternion = new THREE.Quaternion();
            quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), tangent);
            cell.mesh.quaternion.copy(quaternion);
            const scale = THREE.MathUtils.lerp(0.9, 0.7, latestRiskScore) + Math.sin((t + time) * 10) * 0.03;
            cell.mesh.scale.set(scale, scale, scale);
            const baseColor = t <= track.arterialStartT ? inBase : outBase;
            let shaded = baseColor.clone().lerp(riskTint, latestRiskScore * 0.5);

            // Factor-specific color tweaks
            if (lifestyleFactors.cholesterol) {
                // Slightly more yellow and dense when cholesterol is high
                shaded.offsetHSL(0.04, 0.12, -0.05);
            }
            if (lifestyleFactors.smoking) {
                // Darken and desaturate to suggest "smoky" thick blood
                shaded.offsetHSL(-0.03, -0.08, -0.08);
            }
            if (lifestyleFactors.bp) {
                // Sharper, slightly brighter tone for high-pressure flow
                shaded.offsetHSL(0.0, 0.05, 0.03);
            }
            if (lifestyleFactors.alcohol) {
                // Add a subtle brightness shift for alcohol-related irregular flow
                shaded.offsetHSL(0.0, 0.0, 0.04);
            }
            cell.mesh.material.color.copy(shaded);

            // Emissive glow responds to risk and adds flicker with alcohol use
            let emissiveTint = riskTint.clone().multiplyScalar(0.3 + latestRiskScore * 0.6);
            if (lifestyleFactors.alcohol) {
                const flicker = 0.2 + 0.15 * Math.sin((time + cell.offset) * 7.0);
                emissiveTint.multiplyScalar(1 + flicker);
            }
            cell.mesh.material.emissive.copy(emissiveTint);
            if (realismEnabled && (lifestyleFactors.smoking || lifestyleFactors.cholesterol)) {
                let nearPlaque = false;
                for (let i = 0; i < plaquePositions.length; i++) {
                    if (point.distanceTo(plaquePositions[i]) < 2.8) {
                        nearPlaque = true;
                        break;
                    }
                }
                if (nearPlaque) {
                    const jitter = (Math.sin(time * 12 + cell.offset * 20) * 0.08);
                    cell.mesh.position.add(new THREE.Vector3(jitter, -jitter * 0.5, jitter * 0.6));
                }
            }
        });
    });
}

function setBloodFlowVisibility(enabled) {
    bloodFlowTracks.forEach((track) => {
        const showLines = enabled && !realismEnabled;
        track.lineIn.visible = showLines;
        track.lineOut.visible = showLines;
        track.cells.forEach((cell) => {
            cell.mesh.visible = enabled;
        });
    });
}

function handleProgressionInput() {
	progressionValue = Number(progressionSlider.value);
	updateRiskAndProgression();
}

function updateRiskAndProgression() {
	const riskScore = calculateRiskScore();
	latestRiskScore = riskScore;
	if (riskIndicatorValue) {
		riskIndicatorValue.textContent = `${Math.round(riskScore * 100)}%`;
		const color = riskScore > 0.66 ? '#FF4C4C' : riskScore > 0.33 ? '#FFD700' : '#7CFC00';
		riskIndicatorValue.style.color = color;
	}
	applyProgressionVisuals(progressionValue / 100, riskScore);
	updateBloodFlowVisual(riskScore);
	updateRiskDetailMessage(riskScore);
	updateDiseaseOverlayIntensity(riskScore);
	createHealthFactorIndicators(); // Update health factor indicators
}

function calculateRiskScore() {
	let score = progressionValue / 100;
	if (lifestyleFactors.smoking) {
		score += 0.2;
	}
	if (lifestyleFactors.bp) {
		score += 0.15;
	}
	if (lifestyleFactors.cholesterol) {
		score += 0.15;
	}
	if (lifestyleFactors.alcohol) {
		score += 0.1;
	}
	if (!lifestyleFactors.exercise) {
		score += 0.1;
	} else {
		score -= 0.1;
	}
	return THREE.MathUtils.clamp(score, 0, 1);
}

function updateRiskDetailMessage(riskScore) {
	if (!riskIndicatorDetail) {
		return;
	}
	const factors = [];
	if (lifestyleFactors.smoking) {
		factors.push('Smoking thickens artery walls and slows oxygen delivery.');
	}
	if (lifestyleFactors.bp) {
		factors.push('High blood pressure stresses vessel linings and accelerates plaque.');
	}
	if (lifestyleFactors.cholesterol) {
		factors.push('High cholesterol deposits waxy plaque inside arteries.');
	}
	if (lifestyleFactors.alcohol) {
		factors.push('Excess alcohol weakens muscle contractions and disrupts rhythm.');
	}
	if (!lifestyleFactors.exercise) {
		factors.push('Sedentary habits slow circulation and impair recovery.');
	}
	if (!factors.length) {
		riskIndicatorDetail.textContent = 'Habits are supporting healthy flow.';
		return;
	}
	const primary = factors[0];
	const extra = factors.length > 1 ? ` (+${factors.length - 1} additional factor${factors.length - 1 > 1 ? 's' : ''})` : '';
	const severity = riskScore > 0.7 ? 'Critical flow restriction risk.' : riskScore > 0.4 ? 'Moderate strain on circulation.' : 'Mild impact on circulation.';
}

function applyProgressionVisuals(progress, riskScore) {
    // If everything is truly healthy (no disease, no adverse lifestyle factor,
    // no progression), keep the emissive exactly at the original base value so
    // the heart appears in its natural light-pink color.
    const hasLifestyleFactor =
        lifestyleFactors.smoking ||
        lifestyleFactors.bp ||
        lifestyleFactors.cholesterol ||
        lifestyleFactors.alcohol ||
        !lifestyleFactors.exercise;
    const hasDisease = currentDiseaseState !== 'healthy';
    if (!hasLifestyleFactor && !hasDisease && riskScore <= 0 && progress <= 0) {
        heartMeshes.forEach((mesh) => {
            if (mesh.userData.baseEmissive) {
                mesh.material.emissive.copy(mesh.userData.baseEmissive);
            } else {
                mesh.material.emissive.set(0x000000);
            }
            if (mesh.material.emissiveIntensity !== undefined) {
                // Only apply emissive intensity for diseased hearts
                mesh.material.emissiveIntensity = 0;
            }
        });
        return;
    }

    const riskColor = new THREE.Color(0x440000);
    heartMeshes.forEach((mesh) => {
        const baseEmissive = mesh.userData.baseEmissive || new THREE.Color(0x000000);
        const emissiveOverlay = baseEmissive.clone().lerp(riskColor, progress);
        mesh.material.emissive.copy(emissiveOverlay);
        if (mesh.material.emissiveIntensity !== undefined) {
            // Only apply emissive intensity for diseased hearts
            mesh.material.emissiveIntensity = riskScore > 0 ? 0.6 + riskScore * 0.7 : 0;
        }
    });
}

function createDiseaseOverlays() {
    const overlayRoot = new THREE.Group();
    overlayRoot.name = 'diseaseOverlaysRoot';
    scene.add(overlayRoot);

    const plaqueMaterial = new THREE.MeshStandardMaterial({
        color: 0xe6a94a,
        emissive: 0x5c2d00,
        transparent: true,
        opacity: 0.85,
        emissiveIntensity: 1.6
    });
	const plaqueGroup = new THREE.Group();
	plaqueGroup.visible = false;
	plaquePositions.forEach((pos, index) => {
		const geo = new THREE.SphereGeometry(1.15 - index * 0.15, 32, 32);
		const mesh = new THREE.Mesh(geo, plaqueMaterial.clone());
		mesh.position.copy(pos);
		mesh.userData.baseOpacity = mesh.material.opacity;
		plaqueGroup.add(mesh);
	});
	overlayRoot.add(plaqueGroup);
	diseaseOverlays.atherosclerosis = plaqueGroup;

    const valveMaterial = new THREE.MeshStandardMaterial({
        color: 0xe0b85a,
        emissive: 0xcc7a00,
        transparent: true,
        opacity: 0.85,
        emissiveIntensity: 1.7
    });
	const valveTorus = new THREE.Mesh(new THREE.TorusGeometry(1.5, 0.35, 24, 48), valveMaterial);
	valveTorus.position.copy(valvePosition);
	valveTorus.rotation.set(Math.PI / 3, 0.2, 0);
	valveTorus.userData.baseOpacity = valveMaterial.opacity;
	const valveGroup = new THREE.Group();
	valveGroup.visible = false;
	valveGroup.add(valveTorus);
	overlayRoot.add(valveGroup);
	diseaseOverlays.valve_stenosis = valveGroup;

    const infarctMaterial = new THREE.MeshStandardMaterial({
        color: 0xb00020,
        emissive: 0x8b001a,
        transparent: true,
        opacity: 0.8,
        emissiveIntensity: 1.8
    });
	const infarctSphere = new THREE.Mesh(new THREE.SphereGeometry(2, 32, 32), infarctMaterial);
	infarctSphere.position.copy(infarctPosition);
	infarctSphere.userData.baseOpacity = infarctMaterial.opacity;
	infarctSphere.userData.pulse = true;
	const infarctGroup = new THREE.Group();
	infarctGroup.visible = false;
	infarctGroup.add(infarctSphere);
    overlayRoot.add(infarctGroup);
    diseaseOverlays.myocardial_infarction = infarctGroup;

    focusArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 2.6, 0xfff9c4, 0.8, 0.45);
	focusArrow.cone.material.transparent = true;
	focusArrow.line.material.transparent = true;
	focusArrow.cone.material.opacity = 0;
	focusArrow.line.material.opacity = 0;
	focusArrow.visible = false;
	focusArrow.renderOrder = 13;
    overlayRoot.add(focusArrow);
}

function activateDiseaseOverlay(stateName) {
    if (activeOverlayGroup) {
        activeOverlayGroup.visible = false;
    }
    activeOverlayGroup = diseaseOverlays[stateName] || null;
    if (activeOverlayGroup) {
        activeOverlayGroup.visible = true;
    }
    const focus = diseaseFocus[stateName];
}

function updateDiseaseOverlayIntensity(riskScore) {
    const plaqueGroup = diseaseOverlays.atherosclerosis;
    const valveGroup = diseaseOverlays.valve_stenosis;
    const infarctGroup = diseaseOverlays.myocardial_infarction;

    // Strengthen plaque deposits when cholesterol / smoking / high BP are active,
    // even if the global disease state is not explicitly set to atherosclerosis.
    const plaqueFactors = [
        lifestyleFactors.cholesterol,
        lifestyleFactors.smoking,
        lifestyleFactors.bp,
    ].filter(Boolean).length;

    if (plaqueGroup) {
        // Show plaque either when disease state is atherosclerosis or when at least
        // one plaque-related lifestyle factor is enabled.
        plaqueGroup.visible = (currentDiseaseState === 'atherosclerosis') || plaqueFactors > 0;
        const extra = plaqueFactors > 0 ? 0.2 + 0.15 * (plaqueFactors - 1) : 0;
        const plaqueRisk = THREE.MathUtils.clamp(riskScore + extra, 0, 1);
        plaqueGroup.children.forEach((mesh) => {
            const baseOpacity = mesh.userData.baseOpacity || 0.5;
            mesh.material.opacity = THREE.MathUtils.clamp(baseOpacity + plaqueRisk * 0.6, 0, 1);
            if (mesh.material.emissiveIntensity !== undefined) {
                mesh.material.emissiveIntensity = 1.3 + plaqueRisk * 1.6;
            }
        });
    }

    // For other disease overlays (valve stenosis, infarction), keep using the
    // base risk score so they still respond to progression and disease state.
    Object.entries(diseaseOverlays).forEach(([name, group]) => {
        if (!group || name === 'atherosclerosis') {
            return;
        }
        group.children.forEach((mesh) => {
            const baseOpacity = mesh.userData.baseOpacity || 0.5;
            mesh.material.opacity = THREE.MathUtils.clamp(baseOpacity + riskScore * 0.5, 0, 1);
            if (mesh.material.emissiveIntensity !== undefined) {
                mesh.material.emissiveIntensity = 1.2 + riskScore * 1.4;
            }
        });
    });
}

function updateOverlayPulse() {
    if (!activeOverlayGroup) {
        return;
    }
    const time = performance.now() * 0.001;
    const pulse = 1 + Math.sin(time * 2) * 0.14 * (0.4 + latestRiskScore);
    activeOverlayGroup.children.forEach((mesh) => {
        mesh.scale.set(pulse, pulse, pulse);
    });
}

function startGuidedTour() {
	if (guidedTourActive) {
		return;
	}
	guidedTourActive = true;
	controls.enabled = false;
	executeTourStep(0);
}

function executeTourStep(index) {
	if (index >= tourStops.length) {
		guidedTourActive = false;
		controls.enabled = true;
		hideAnnotationPanel();
		return;
	}
	const stop = tourStops[index];
	showAnnotationPanel(stop.hotspotIndex);
	annotationPanel.title.textContent = stop.title;
	annotationPanel.description.textContent = `${annotationPanel.description.textContent}\n${stop.narration}`;
	const duration = 3500;
	const cameraState = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
	const targetState = { x: controls.target.x, y: controls.target.y, z: controls.target.z };
	let completedTweens = 0;
	const handleComplete = () => {
		completedTweens += 1;
		if (completedTweens === 2) {
			setTimeout(() => executeTourStep(index + 1), 1200);
		}
	};
	new TWEEN.Tween(cameraState)
		.to(stop.camera, duration)
		.easing(TWEEN.Easing.Cubic.InOut)
		.onUpdate(() => {
			camera.position.set(cameraState.x, cameraState.y, cameraState.z);
		})
		.onComplete(handleComplete)
		.start();
	new TWEEN.Tween(targetState)
		.to(stop.target, duration)
		.easing(TWEEN.Easing.Cubic.InOut)
		.onUpdate(() => {
			controls.target.set(targetState.x, targetState.y, targetState.z);
		})
		.onComplete(handleComplete)
		.start();
}

const HOTSPOT_RADIUS = 15;
const tourStops = [
	{
		title: 'Aorta Overview',
		hotspotIndex: 0,
		camera: { x: -55, y: 18, z: 15 },
		target: { x: 0, y: 5, z: 0 },
		narration: 'The aorta distributes oxygen-rich blood. Plaque here limits flow to vital organs.'
	},
	{
		title: 'Pulmonary Circuit',
		hotspotIndex: 4,
		camera: { x: 20, y: 25, z: 35 },
		target: { x: 5, y: 5, z: 5 },
		narration: 'Pulmonary veins return oxygenated blood. Scarring impairs filling efficiency.'
	},
	{
		title: 'Vena Cava Checkpoint',
		hotspotIndex: 5,
		camera: { x: -10, y: 35, z: -20 },
		target: { x: -5, y: 5, z: 0 },
		narration: 'Obstruction of the vena cava leads to swelling and reduced venous return.'
	}
];

const diseaseStates = {
    healthy: {
        label: 'Healthy',
        color: 0xCC3333,
        scale: 1,
    },
    atherosclerosis: {
        label: 'Atherosclerosis',
        color: 0x8F6A2B,
        scale: 1.05,
    },
    valve_stenosis: {
        label: 'Valve Stenosis',
        color: 0xC97F1A,
        scale: 0.97,
    },
    myocardial_infarction: {
        label: 'Myocardial Infarction',
        color: 0x4B0000,
        scale: 1.1,
    }
};

const diseaseFocus = {
	atherosclerosis: {
		hotspotIndex: 0,
		title: 'Aorta — Plaque Buildup',
		description: 'Plaque gradually lines the aortic wall, narrowing the channel that feeds every organ.',
		disease: 'Untreated plaque restricts blood flow and can trigger organ ischemia or a full blockage.',
		camera: { x: -55, y: 18, z: 15 },
		target: { x: 0, y: 5, z: 0 }
	},
	valve_stenosis: {
		hotspotIndex: 2,
		title: 'Valve Leaflets — Stenosis',
		description: 'Calcified valve leaflets struggle to open, lowering forward blood flow.',
		disease: 'Pressure builds behind the valve, forcing the heart muscle to overwork and eventually weaken.',
		camera: { x: -20, y: 10, z: 32 },
		target: { x: -3, y: 4, z: 5 }
	},
	myocardial_infarction: {
		hotspotIndex: 3,
		title: 'Muscle Wall — Infarction',
		description: 'A blocked coronary artery starves nearby myocardium of oxygen.',
		disease: 'Heart muscle cells die within minutes, reducing pump strength and risking lethal arrhythmias.',
		camera: { x: 12, y: 22, z: 28 },
		target: { x: 4, y: 6, z: 8 }
	}
};

function handleHotspotClick(event) {
	const canvas = document.getElementById("dotsCanvas");
	const rect = canvas.getBoundingClientRect();
	const clickX = event.clientX - rect.left;
	const clickY = event.clientY - rect.top;

	for (let i = 0; i < hotspotScreenPositions.length; i++) {
		const hotspot = hotspotScreenPositions[i];
		if (!hotspot) continue;
		const dx = hotspot.x - clickX;
		const dy = hotspot.y - clickY;
		const distance = Math.sqrt(dx * dx + dy * dy);
		if (distance <= HOTSPOT_RADIUS) {
			showAnnotationPanel(i);
			return;
		}
	}
	if (camera && controls) {
		focusOnDiseaseState(currentDiseaseState);
	}
	activateDiseaseOverlay(currentDiseaseState);
	applyProgressionVisuals(progressionValue / 100, calculateRiskScore());
}

function showAnnotationPanel(index) {
	const data = annotationContent[index];
	if (!data) {
		return;
	}
	activeHotspotIndex = index;
	annotationPanel.title.textContent = data.title;
	annotationPanel.description.textContent = data.description;
	annotationPanel.disease.textContent = data.disease;
	annotationPanel.container.classList.remove('hidden');
	// redraw dots to update highlight color
	checkDots();
}

function hideAnnotationPanel() {
	annotationPanel.container.classList.add('hidden');
	activeHotspotIndex = null;
	checkDots();
}

function focusOnDiseaseState(stateName) {
	const focus = diseaseFocus[stateName];
	if (!focus) {
		return;
	}
	if (typeof focus.hotspotIndex === 'number') {
		showAnnotationPanel(focus.hotspotIndex);
	}
	annotationPanel.title.textContent = focus.title;
	annotationPanel.description.textContent = focus.description;
	annotationPanel.disease.textContent = focus.disease;
	annotationPanel.container.classList.remove('hidden');
	if (!camera || !controls) {
		return;
	}
	smoothFocusCamera(focus.camera, focus.target);
    showFocusIndicator(focus.target);
    currentFocusTarget = new THREE.Vector3(focus.target.x, focus.target.y, focus.target.z);
    computeFocusPointT();
}

function focusOnLifestyleFactor(name) {
    const target = lifestyleFocusTargets[name];
    if (!target || !camera || !controls) {
        return;
    }
    const cam = camera.position.clone();
    const cameraTarget = { x: cam.x + (target.x - controls.target.x), y: cam.y + (target.y - controls.target.y), z: cam.z + (target.z - controls.target.z) };
    smoothFocusCamera(cameraTarget, target);
    showFocusIndicator(target);
    currentFocusTarget = new THREE.Vector3(target.x, target.y, target.z);
    computeFocusPointT();
}

function computeFocusPointT() {
    if (!currentFocusTarget || !bloodFlowTracks.length) {
        focusPointT = null;
        return;
    }
    const curve = bloodFlowTracks[0].curve;
    const samples = 240;
    let minD = Infinity;
    let minU = 0;
    for (let i = 0; i <= samples; i++) {
        const u = i / samples;
        const p = curve.getPointAt(u);
        const d = p.distanceTo(currentFocusTarget);
        if (d < minD) {
            minD = d;
            minU = u;
        }
    }
    focusPointT = minU;
}

function smoothFocusCamera(cameraTarget, controlTarget) {
	if (!cameraTarget || !controlTarget) {
		return;
	}
	const duration = 1600;
	// prevent zoom-out when realism is enabled by clamping new camera distance
	const currentDist = camera.position.clone().sub(controls.target).length();
	const targetVec = new THREE.Vector3(controlTarget.x, controlTarget.y, controlTarget.z);
	const desiredCam = new THREE.Vector3(cameraTarget.x, cameraTarget.y, cameraTarget.z);
	const desiredDist = desiredCam.clone().sub(targetVec).length();
	if (realismEnabled && desiredDist > currentDist) {
		const dir = desiredCam.clone().sub(targetVec).normalize();
		const newDist = currentDist * 0.9; // slight zoom-in
		const clampedCam = targetVec.clone().add(dir.multiplyScalar(Math.max(0.1, newDist)));
		cameraTarget = { x: clampedCam.x, y: clampedCam.y, z: clampedCam.z };
	}
	const cameraState = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
	const targetState = { x: controls.target.x, y: controls.target.y, z: controls.target.z };
	new TWEEN.Tween(cameraState)
		.to(cameraTarget, duration)
		.easing(TWEEN.Easing.Cubic.InOut)
		.onUpdate(() => {
			camera.position.set(cameraState.x, cameraState.y, cameraState.z);
		})
		.start();
	new TWEEN.Tween(targetState)
		.to(controlTarget, duration)
		.easing(TWEEN.Easing.Cubic.InOut)
		.onUpdate(() => {
			controls.target.set(targetState.x, targetState.y, targetState.z);
		})
		.start();
}

function showFocusIndicator(target) {
	if (!focusArrow || !target || !camera) {
		return;
	}
	const dir = camera.position.clone().sub(target).normalize();
	const arrowOrigin = target.clone().add(dir.clone().multiplyScalar(2.6));
	focusArrow.position.copy(arrowOrigin);
	const arrowDir = target.clone().sub(arrowOrigin).normalize();
	focusArrow.setDirection(arrowDir);
	focusArrow.visible = true;
	const tweenObj = { opacity: 0 };
	focusArrow.cone.material.opacity = 0;
	focusArrow.line.material.opacity = 0;

	new TWEEN.Tween(tweenObj)
		.to({ opacity: 0.85 }, 700)
		.easing(TWEEN.Easing.Cubic.Out)
		.onUpdate(() => {
			focusArrow.cone.material.opacity = tweenObj.opacity;
			focusArrow.line.material.opacity = tweenObj.opacity * 0.7;
		})
		.onComplete(() => {
			new TWEEN.Tween(tweenObj)
				.to({ opacity: 0 }, 600)
				.onUpdate(() => {
					focusArrow.cone.material.opacity = tweenObj.opacity;
					focusArrow.line.material.opacity = tweenObj.opacity * 0.7;
				})
				.onComplete(() => {
					focusArrow.visible = false;
				})
				.start();
		})
		.start();
}

// set up the canvas on which to draw the locator line
// MUST NOT be below init() and animate()
var lineCanvas = document.createElement( 'canvas' );
lineCanvas.id = "dotsCanvas";
lineCanvas.width = window.innerWidth;
lineCanvas.height = window.innerHeight;
lineCanvas.style.position = "absolute";
lineCanvas.style.zIndex = "101";
lineCanvas.style.display = "inline";
appContainer.appendChild(lineCanvas);
lineCanvas.addEventListener('click', handleHotspotClick);
annotationPanel.closeBtn.addEventListener('click', hideAnnotationPanel);

// creating pin vectors
const allMyPins = [
	new THREE.Vector3(-4.044027262032927, 4.219614694564475, -1.2308631918325883),
	new THREE.Vector3(-3.4000000000000004, 7.600000000000003, 2.5999999999999996),
	new THREE.Vector3(-4.200000000000001, 0.7999999999999987, 3.4000000000000004),
	new THREE.Vector3(-5.000000000000002, 3.1999999999999993, 1.3999999999999995),
	new THREE.Vector3(1.0000000000000004, 2.3999999999999986, 6.000000000000002),
	new THREE.Vector3(-2.1999999999999993, 6.000000000000002, -4.800000000000003)
];

init();
animate();

function init() {
	
	// camera
	camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 2000 );
	camera.position.x= -42.54404597864378; camera.position.y= 4.095160583665006; camera.position.z= -1.2896749071128895;
	
	// scene
	scene = new THREE.Scene();
	let ambient = new THREE.AmbientLight( 0x444444 );
	scene.add(ambient);
	let directionalLight1 = new THREE.DirectionalLight( 0xffeedd, 0.7 );
	directionalLight1.position.set( 0, 0, 1 ).normalize();
	let directionalLight2 = new THREE.DirectionalLight( 0xffeedd, 0.7 );
	directionalLight2.position.set( 0, 0, -1 ).normalize();
	let directionalLight3 = new THREE.DirectionalLight( 0xffeedd, 0.7 );
	directionalLight3.position.set( 1, 0, 0).normalize();
	let directionalLight4 = new THREE.DirectionalLight( 0xffeedd, 0.7 );
	directionalLight4.position.set( -1, 0, 0 ).normalize();
	let directionalLight5 = new THREE.DirectionalLight( 0xffeedd, 0.7 );
	directionalLight5.position.set( 0, 1, 0 ).normalize();
	let directionalLight6 = new THREE.DirectionalLight( 0xffeedd, 0.7 );
	directionalLight6.position.set( 0, -1, 0 ).normalize();
	scene.add( directionalLight1 );
	scene.add( directionalLight2 );
	scene.add( directionalLight3 );
	scene.add( directionalLight4 );			
	scene.add( directionalLight5 );
	scene.add( directionalLight6 );
	createDiseaseOverlays();
	
	// models
	const onProgress = function (xhr) {
		if (xhr.lengthComputable) {
			const percentComplete = xhr.loaded / xhr.total * 100;
			console.log( Math.round(percentComplete, 2) + '% downloaded' );
		}
	};

function registerHeartMeshes(rootObject) {
	if (!rootObject) {
		return;
	}
	rootObject.traverse((child) => {
		if (child.isMesh) {
			if (!child.material.isMaterial) {
				return;
			}
			child.material = child.material.clone();
			child.userData.baseColor = child.material.color.clone();
			if (!child.material.emissive) {
				child.material.emissive = new THREE.Color(0x000000);
			}
			child.userData.baseEmissive = child.material.emissive.clone();
			child.userData.baseScale = child.scale.clone();
			if (!heartMeshes.includes(child)) {
				heartMeshes.push(child);
			}
		}
	});
}

function setDiseaseState(stateName) {
	const config = diseaseStates[stateName] || diseaseStates.healthy;
	currentDiseaseState = stateName in diseaseStates ? stateName : 'healthy';
	const scaleTarget = config.scale || 1;
	const openHeart = scene?.getObjectByName('openHeart', true);
	activateDiseaseOverlay(currentDiseaseState);
	focusOnDiseaseState(currentDiseaseState);

	// Keep the main heart meshes in their original material colors; disease is
	// represented via dedicated overlay meshes (plaques, valve ring, infarct)
	// and risk-based emissive effects rather than fully recoloring the heart.
	heartMeshes.forEach((mesh) => {
		if (mesh.userData.baseColor) {
			mesh.material.color.copy(mesh.userData.baseColor);
		}
		if (mesh.userData.baseEmissive) {
			mesh.material.emissive.copy(mesh.userData.baseEmissive);
		}
	});

	if (openHeart) {
		const startScale = {
			x: openHeart.scale.x,
			y: openHeart.scale.y,
			z: openHeart.scale.z
		};
		new TWEEN.Tween(startScale)
			.to({ x: scaleTarget, y: scaleTarget, z: scaleTarget }, 800)
			.onUpdate(() => {
				openHeart.scale.set(startScale.x, startScale.y, startScale.z);
			})
			.start();
		if (!openHeart.userData.baseScale) {
			openHeart.userData.baseScale = new THREE.Vector3(openHeart.scale.x, openHeart.scale.y, openHeart.scale.z);
		}
	}
	
	// Update rhythm indicator when disease state changes
	updateRiskAndProgression();
}

	const onError = function () { };
	
	const manager = new THREE.LoadingManager();
	manager.addHandler( /\.dds$/i, new DDSLoader() );

	new MTLLoader(manager)
		.setPath(BASE_URL)
		.load( 'openheartLD1.mtl', function (materials) {
			materials.preload();
			new OBJLoader(manager)
				.setMaterials(materials)
				.setPath(BASE_URL)
					.load( 'openheartLD1.obj', function (object) {
					object.position.x = 32;
					object.position.y = -3;
					object.position.z = 3;
					object.rotation.x = -1.6;
					object.name = 'openHeart';
					scene.add(object);
					flowRoot = new THREE.Group();
					flowRoot.name = 'flowRoot';
					object.add(flowRoot);
					registerHeartMeshes(object);
					setDiseaseState(currentDiseaseState);
					const dsSelect = document.getElementById("disease_state_select");
				}, onProgress, onError );
		});

	new MTLLoader(manager)
		.setPath(BASE_URL)
		.load( 'heartpartLD1.mtl', function (materials) {
			materials.preload();
			new OBJLoader(manager)
				.setMaterials(materials)
				.setPath(BASE_URL)
				.load( 'heartpartLD1.obj', function (object) {
					object.position.x= -7.600000000000003; object.position.y= -9.399999999999999; object.position.z= 24.199999999999946;
					object.rotation.x= -1.6000000000000008; object.rotation.y= 0.44999999999999996; object.rotation.z= 0.05000000000000002;
					object.scale.x= 3.949999999999994; object.scale.y= 3.949999999999994; object.scale.z= 3.949999999999994;
					object.name = 'heartPart';
					scene.add(object);
					heartPart = object;
					if (openHeartRequested) {
						openHeartRequested = false;
						movePart();
					}
				}, onProgress, onError );
		});

	heartPart = scene.getObjectByName('heartPart', true);
	heartIsOpen = false;

	// Apply initial model configuration if provided (e.g., "openheart" to
	// start with the heart opened rather than closed).
	if (initialConfig.model === 'openheart') {
		openHeartRequested = true;
	}

	camera.position.z = 5;
	
	// renderer three.js in canvas
	var threeCanvas = document.getElementById('threeCanvas');
	renderer = new THREE.WebGLRenderer({canvas: threeCanvas});
	threeCanvas.width  = window.innerWidth;
	threeCanvas.height = window.innerHeight;
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( window.innerWidth, window.innerHeight );
	
	// window resize
	window.addEventListener( 'resize', onWindowResize, false );

	raycaster = new THREE.Raycaster();
	mouse = new THREE.Vector2();

	const moveBtn = document.getElementById("move_heart_btn");
	if (moveBtn) {
		moveBtn.addEventListener("click", movePart);
	}
	const hideDotsBtn = document.getElementById("hide_dots_btn");
	if (hideDotsBtn) {
		hideDotsBtn.addEventListener("click", hideDots);
	}
	const diseaseSelect = document.getElementById("disease_state_select");
	if (diseaseSelect) {
		diseaseSelect.addEventListener("change", (event) => {
			setDiseaseState(event.target.value);
		});
	}
	progressionSlider?.addEventListener('input', handleProgressionInput);
	smokingToggle?.addEventListener('change', () => {
    lifestyleFactors.smoking = smokingToggle.checked;
    updateRiskAndProgression();
    if (realismEnabled && smokingToggle.checked) {
        focusOnLifestyleFactor('smoking');
    }
});
	bpToggle?.addEventListener('change', () => {
    lifestyleFactors.bp = bpToggle.checked;
    updateRiskAndProgression();
    if (realismEnabled && bpToggle.checked) {
        focusOnLifestyleFactor('bp');
    }
});
	cholesterolToggle?.addEventListener('change', () => {
    lifestyleFactors.cholesterol = cholesterolToggle.checked;
    updateRiskAndProgression();
    if (realismEnabled && cholesterolToggle.checked) {
        focusOnLifestyleFactor('cholesterol');
    }
});
	alcoholToggle?.addEventListener('change', () => {
    lifestyleFactors.alcohol = alcoholToggle.checked;
    updateRiskAndProgression();
    if (realismEnabled && alcoholToggle.checked) {
        focusOnLifestyleFactor('alcohol');
    }
});
    exerciseToggle?.addEventListener('change', () => {
        lifestyleFactors.exercise = exerciseToggle.checked;
        updateRiskAndProgression();
    });
    realismToggle?.addEventListener('change', () => {
        realismEnabled = realismToggle.checked;
        updateRiskAndProgression();
        if (realismEnabled) {
            focusOnDiseaseState(currentDiseaseState);
            setBloodFlowVisibility(bloodFlowEnabled);
        }
    });
    
    // Add event listener for rhythm toggle
    rhythmToggle?.addEventListener('change', () => {
        rhythmEnabled = rhythmToggle.checked;
        updateRiskAndProgression();
    });
    
    guidedTourBtn?.addEventListener('click', startGuidedTour);
    createControls(camera);
    createBloodFlowIndicator();
    setBloodFlowVisibility(bloodFlowEnabled);
    updateRiskAndProgression();
    createHealthFactorIndicators(); // Create initial indicators
    
	// Initialize UI toggle states based on initial configuration
	if (smokingToggle) {
		smokingToggle.checked = lifestyleFactors.smoking;
	}
	if (bpToggle) {
		bpToggle.checked = lifestyleFactors.bp;
	}
	if (cholesterolToggle) {
		cholesterolToggle.checked = lifestyleFactors.cholesterol;
	}
	if (alcoholToggle) {
		alcoholToggle.checked = lifestyleFactors.alcohol;
	}
	if (exerciseToggle) {
		exerciseToggle.checked = lifestyleFactors.exercise;
	}
	if (bloodFlowToggle) {
		bloodFlowToggle.checked = bloodFlowEnabled;
	}
	if (realismToggle) {
		realismToggle.checked = realismEnabled;
	}
	if (rhythmToggle) {
		rhythmToggle.checked = rhythmEnabled;
	}
}

function onWindowResize() {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth, window.innerHeight );
	controls.handleResize();
	animate();
}

function animate() {
	requestAnimationFrame( animate );
	controls.update();
	TWEEN.update();
	updateOverlayPulse();
	updateHeartbeat();
	updateBloodCellFlow();
	if (document.getElementById("dotsCanvas").style.display == "inline") {
		checkDots();
	}
	render();
}

function render() {
	renderer.render( scene, camera );
}

function checkDots() {
	var vector = new THREE.Vector3();
	var openHeart = scene.getObjectByName('openHeart', true);
	var origin;
	var i;
	var canvas = document.getElementById("dotsCanvas");
	canvas.width = canvas.width;
	vector.set(camera.position.x, camera.position.y, camera.position.z);
	if (openHeart) {
		for (i in allMyPins) {
			origin = allMyPins[i].clone();
			// turns vector into a directional vector pointing from origin to camera
			vector.sub(origin).normalize();
			raycaster.set( origin, vector );
			var intersects = raycaster.intersectObject( openHeart, true );
			if (intersects.length > 0) {
				//console.log("marker invisible!");
			}
			else {
				createDots(origin, i);
			}
			vector.set(camera.position.x, camera.position.y, camera.position.z);
		}
	}
}

function createDots( myVector, i ) {
	var vector = myVector.clone();
	vector.project( camera );
	var canvas = document.getElementById("dotsCanvas");
	vector.x = Math.round( (   vector.x + 1 ) * window.innerWidth  / 2 );
	vector.y = Math.round( ( - vector.y + 1 ) * window.innerHeight / 2 );
	vector.z = 0;
	var ctx = canvas.getContext("2d");
	hotspotScreenPositions[i] = { x: vector.x, y: vector.y };
	ctx.beginPath();
	ctx.arc(vector.x,vector.y,15,0,2*Math.PI);
	ctx.fillStyle = i === activeHotspotIndex ? "#FFD700" : "#FF0000";
	ctx.fill();
	// now add the number to the dot
	i++;
	var text = i;
	var font = "bold " + 16 + "px monospace";
	ctx.font = font;
	var width = ctx.measureText(text).width;
	var height = ctx.measureText("w").width;
	ctx.fillStyle = "white";
	ctx.fillText(text, vector.x - (width/2), vector.y + (height/2));
}

function hideDots() {
	var allDots = document.getElementById("dotsCanvas");
	if (allDots.style.display === "inline") {
		allDots.style.display = "none";
	}
	else {
		allDots.style.display = "inline";
	}
}

function movePart() {
	heartPart = scene.getObjectByName('heartPart', true);
	if (!heartPart) {
		return;
	}
	const openZ = Math.PI / 2;
	const closedZ = 0.05;
	
	const rotateCoords = {
		z: heartIsOpen ? openZ : closedZ,
	};

	const tween1 = new TWEEN.Tween(rotateCoords);
	tween1
		.to({z: heartIsOpen ? closedZ : openZ}, 1000)
		.onUpdate(() => {
			heartPart.rotation.z = rotateCoords.z;
		})
		.onComplete(() => {
			// Toggle open/closed state of the shell
			heartIsOpen = !heartIsOpen;
			// When the heart is opened, restore the open-heart mesh colors to their
			// original appearance instead of keeping the red disease tint.
			if (heartIsOpen) {
				heartMeshes.forEach((mesh) => {
					if (mesh.userData.baseColor) {
						mesh.material.color.copy(mesh.userData.baseColor);
					}
					if (mesh.userData.baseEmissive) {
						mesh.material.emissive.copy(mesh.userData.baseEmissive);
					}
				});
			} else {
				// When closing again, reapply disease state visuals so the outer
				// heart shell reflects current pathology.
				setDiseaseState(currentDiseaseState);
			}
		})
		.start();
}

function createControls(camera) {

	controls = new TrackballControls(camera, appContainer);

	controls.rotateSpeed = 1.0;
	controls.zoomSpeed = 1.2;
	controls.panSpeed = 0.8;

	controls.keys = [ 'KeyA', 'KeyS', 'KeyD' ];
}

// Check if any disease state or lifestyle factor is active
function isAnyFactorActive() {
    // Check if any lifestyle factor is enabled
    const hasLifestyleFactor = lifestyleFactors.smoking || lifestyleFactors.bp || 
                              lifestyleFactors.cholesterol || lifestyleFactors.alcohol || 
                              !lifestyleFactors.exercise;
    
    // Check if disease state is not healthy
    const hasDisease = currentDiseaseState !== 'healthy';
    
    // Return true if rhythm is enabled OR if any factor is active
    return rhythmEnabled || (hasLifestyleFactor || hasDisease || realismEnabled);
}

function updateHeartbeat() {
	// Drive heartbeat when rhythm toggle is enabled, regardless of
	// whether additional risk factors or diseases are active.
	if (!rhythmEnabled) {
		return;
	}
    
    const openHeart = scene.getObjectByName('openHeart', true);
    const closedHeart = heartPart;
    
    // If neither heart model is available, return
    if (!openHeart && !closedHeart) {
        return;
    }
    
    const t = performance.now() * 0.001;
    let amp = 0.02;
    let freq = 1.2;
    
    // Base rhythm changes based on disease state
    switch (currentDiseaseState) {
        case 'atherosclerosis':
            amp = 0.025;
            freq = 1.1;
            break;
        case 'valve_stenosis':
            amp = 0.03;
            freq = 1.0;
            break;
        case 'myocardial_infarction':
            amp = 0.04;
            freq = 0.9;
            break;
    }
    
    // Lifestyle factor modifications
    if (lifestyleFactors.smoking) {
        amp += 0.01;
        freq *= 1.1;
    }
    
    if (lifestyleFactors.bp) {
        amp = 0.035;
        freq = 1.6;
    }
    
    if (lifestyleFactors.cholesterol) {
        amp += 0.005;
        freq *= 1.05;
    }
    
    if (lifestyleFactors.alcohol) {
        amp *= 0.75;
        freq *= 0.9 + Math.sin(t * 0.7) * 0.05; // Irregular rhythm
    }
    
    if (!lifestyleFactors.exercise) {
        amp *= 0.8;
        freq *= 0.95;
    }
    
    // Apply realism effects if enabled
    if (realismEnabled) {
        // Additional realism effects can be added here
    }
    
    // Add subtle variations to make the rhythm more lifelike
    const variation = 0.002 * Math.sin(t * 0.5);
    const pulse = 1 + Math.sin(t * freq) * (amp + variation);
    
    // Apply rhythm only to the currently visible heart model
    // If heart is open, apply to openHeart; otherwise apply to closedHeart
    if (heartIsOpen && openHeart) {
        const base = openHeart.userData.baseScale || openHeart.scale;
        openHeart.scale.set(base.x * pulse, base.y * pulse, base.z * pulse);
    } else if (!heartIsOpen && closedHeart) {
        const base = closedHeart.userData.baseScale || closedHeart.scale;
        closedHeart.scale.set(base.x * pulse, base.y * pulse, base.z * pulse);
    }
}