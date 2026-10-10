/* MoleculeIQ front-end. Vanilla JS.
 *
 * Security: every server- or PubChem-provided string is inserted with textContent / text nodes
 * (via the `h()` helper). Do not use innerHTML, outerHTML or insertAdjacentHTML in this file.
 */
(() => {
    'use strict';

    const SVG_NS = 'http://www.w3.org/2000/svg';
    const REQUEST_TIMEOUT_MS = 90000;
    const CLAMP_CHARS = 260;

    const form = document.getElementById('analyze-form');
    const queryInput = document.getElementById('query');
    const queryLabel = document.getElementById('query-label');
    const queryError = document.getElementById('query-error');
    const submitBtn = document.getElementById('submit');
    const feedback = document.getElementById('feedback');
    const results = document.getElementById('results');
    const examplesBox = document.getElementById('examples');

    const EXAMPLES = {
        name: ['aspirin', 'caffeine', 'ibuprofen'],
        smiles: [
            ['CCO', 'ethanol'],
            ['c1ccccc1', 'benzene'],
            ['CC(=O)Oc1ccccc1C(=O)O', 'aspirin'],
        ],
    };
    const MODE_TEXT = {
        name: { label: 'Chemical name', placeholder: 'e.g. aspirin, caffeine, ibuprofen' },
        smiles: { label: 'SMILES string', placeholder: 'e.g. CC(=O)Oc1ccccc1C(=O)O' },
    };

    let busy = false;

    /* ───────── DOM helpers ───────── */
    function h(tag, props = {}, ...children) {
        const node = document.createElement(tag);
        for (const [key, value] of Object.entries(props)) {
            if (value === null || value === undefined || value === false) continue;
            if (key === 'class') node.className = value;
            else if (key === 'text') node.textContent = value;
            else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
            else node.setAttribute(key, value === true ? '' : value);
        }
        for (const child of children.flat()) {
            if (child !== null && child !== undefined && child !== false) node.append(child);
        }
        return node;
    }

    function icon(name, extraClass = '') {
        const svg = document.createElementNS(SVG_NS, 'svg');
        svg.setAttribute('class', `icon ${extraClass}`.trim());
        svg.setAttribute('aria-hidden', 'true');
        const use = document.createElementNS(SVG_NS, 'use');
        use.setAttribute('href', `#i-${name}`);
        svg.append(use);
        return svg;
    }

    function svgEl(tag, attrs = {}) {
        const node = document.createElementNS(SVG_NS, tag);
        for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
        return node;
    }

    function clear(node) { node.replaceChildren(); }

    /* ───────── Clipboard ───────── */
    async function copyText(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (_) {
            const area = h('textarea', { 'aria-hidden': 'true', tabindex: '-1', style: 'position:fixed;left:-9999px;top:0' });
            area.value = text;
            document.body.append(area);
            area.select();
            let ok = false;
            try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
            area.remove();
            return ok;
        }
    }

    function copyButton(value, label) {
        const btn = h('button', { type: 'button', class: 'icon-btn', 'aria-label': `Copy ${label}`, title: `Copy ${label}` }, icon('copy'));
        let timer = null;
        btn.addEventListener('click', async () => {
            const ok = await copyText(String(value));
            clearTimeout(timer);
            btn.classList.toggle('is-copied', ok);
            btn.replaceChildren(icon(ok ? 'check' : 'alert'));
            btn.setAttribute('aria-label', ok ? `${label} copied` : `Could not copy ${label}`);
            timer = setTimeout(() => {
                btn.classList.remove('is-copied');
                btn.replaceChildren(icon('copy'));
                btn.setAttribute('aria-label', `Copy ${label}`);
            }, 1600);
        });
        return btn;
    }

    function copyField(label, valueNode, rawValue, copyLabel) {
        return h('div', { class: 'ident' },
            h('div', { class: 'ident__label', text: label }),
            h('div', { class: 'copyfield' },
                h('div', { class: 'copyfield__value' }, valueNode),
                copyButton(rawValue, copyLabel)));
    }

    /* ───────── Formatting ───────── */
    function formulaNode(formula) {
        const wrap = document.createDocumentFragment();
        for (const part of String(formula).split(/(\d+)/)) {
            if (!part) continue;
            wrap.append(/^\d+$/.test(part) ? h('sub', { text: part }) : part);
        }
        return wrap;
    }

    function formatScore(value) {
        if (value === 0) return '0';
        return Math.abs(value) >= 0.001 ? value.toFixed(3) : value.toExponential(2);
    }

    function pubchemUrl(cid) {
        return /^\d+$/.test(String(cid)) ? `https://pubchem.ncbi.nlm.nih.gov/compound/${String(cid)}` : null;
    }

    /* ───────── Cards ───────── */
    function card({ title, iconName, source, modifier = '', wide = false, span = '' }, ...body) {
        const tag = source === 'model'
            ? h('span', { class: 'tag tag--model', text: 'Model output' })
            : source === 'pubchem'
                ? h('span', { class: 'tag tag--pubchem', text: 'Source: PubChem' })
                : null;
        return h('article', { class: `card ${modifier} ${wide ? 'card--wide' : ''} ${span}`.replace(/\s+/g, ' ').trim() },
            h('div', { class: 'card__head' },
                h('h3', { class: 'card__title' }, icon(iconName), title),
                tag),
            ...body);
    }

    function emptyState(message) {
        return h('div', { class: 'empty' }, icon('info'), h('p', { text: message }));
    }

    function pubchemFoot(compound) {
        const url = compound && pubchemUrl(compound.cid);
        if (!url) return null;
        return h('div', { class: 'card__foot' }, 'Data from ',
            h('a', { href: url, target: '_blank', rel: 'noopener noreferrer', text: 'PubChem' }),
            ` (CID ${compound.cid})`);
    }

    function textList(items) {
        return h('ul', { class: 'textlist' }, items.map((text, index) => {
            const long = text.length > CLAMP_CHARS;
            const id = `t-${Math.random().toString(36).slice(2, 8)}-${index}`;
            const body = h('p', { class: 'textitem__text', id, text });
            const li = h('li', { class: `textitem${long ? ' is-clamped' : ''}` }, body);
            if (long) {
                const toggle = h('button', { type: 'button', class: 'link-btn', 'aria-expanded': 'false', 'aria-controls': id },
                    h('span', { text: 'Show more' }), icon('chevron'));
                toggle.addEventListener('click', () => {
                    const open = li.classList.toggle('is-clamped') === false;
                    toggle.setAttribute('aria-expanded', String(open));
                    toggle.firstChild.textContent = open ? 'Show less' : 'Show more';
                });
                li.append(toggle);
            }
            return li;
        }));
    }

    /* ───────── Score card ───────── */
    function gaugeSvg(value) {
        const radius = 54, cx = 70, cy = 68;
        const length = Math.PI * radius;
        const d = `M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`;
        const svg = svgEl('svg', { class: 'gauge', viewBox: '0 0 140 84', role: 'img',
            'aria-label': `Raw model output ${value} on a scale from 0 to 1` });
        const defs = svgEl('defs');
        const grad = svgEl('linearGradient', { id: 'gauge-grad', x1: '0', y1: '0', x2: '1', y2: '0' });
        grad.append(svgEl('stop', { offset: '0', 'stop-color': '#4c8dff' }), svgEl('stop', { offset: '1', 'stop-color': '#2dd4e6' }));
        defs.append(grad);
        const track = svgEl('path', { d, class: 'gauge__track' });
        const bar = svgEl('path', { d, class: 'gauge__value' });
        bar.style.strokeDasharray = `0 ${length}`;
        const t0 = svgEl('text', { x: cx - radius, y: cy + 14, 'text-anchor': 'middle', class: 'gauge__tick' });
        t0.textContent = '0';
        const t1 = svgEl('text', { x: cx + radius, y: cy + 14, 'text-anchor': 'middle', class: 'gauge__tick' });
        t1.textContent = '1';
        svg.append(defs, track, bar, t0, t1);
        // Fill is proportional to the raw output; no rescaling, clamping or colour-coding of "good/bad".
        requestAnimationFrame(() => requestAnimationFrame(() => {
            bar.style.strokeDasharray = `${length * value} ${length}`;
        }));
        return svg;
    }

    function scoreCard(data) {
        const raw = typeof data.raw_output === 'number' ? data.raw_output : null;
        const finite = raw !== null && Number.isFinite(raw);
        // The network ends in a sigmoid, so the output is bounded to [0, 1]. If a value ever falls
        // outside that range the arc could not represent it truthfully, so only the number is shown.
        const gaugeSafe = finite && raw >= 0 && raw <= 1;

        const body = h('div', { class: 'score__body' });
        if (gaugeSafe) body.append(gaugeSvg(raw));
        body.append(
            h('div', { class: `score__number${gaugeSafe ? '' : ' score__number--solo'}`, title: finite ? String(raw) : '',
                text: finite ? formatScore(raw) : 'Unavailable' }),
            h('div', { class: 'score__scale', text: finite
                ? (gaugeSafe ? 'Raw network output · scale 0 to 1' : 'Raw network output (outside 0–1; gauge not shown)')
                : 'The server did not return a model output.' }));

        return card({ title: 'ATAD5 stress-response model score', iconName: 'cpu', source: 'model', modifier: 'card--model', span: 'span-5' },
            h('div', {}, h('span', { class: 'tag tag--warn', text: 'Uncalibrated model output' })),
            body,
            h('p', { class: 'score__note' },
                h('strong', { text: 'Not a probability, clinical risk estimate, or validated toxicity measurement. ' }),
                'A low value does not mean a compound is safe, and a high value does not mean it is toxic.'));
    }

    /* ───────── PubChem cards ───────── */
    const PROFILE_ROWS = [
        ['Molecular formula', 'MolecularFormula', '', formulaNode],
        ['Molecular weight', 'MolecularWeight', 'g/mol'],
        ['IUPAC name', 'IUPACName'],
        ['XLogP', 'XLogP'],
        ['Exact mass', 'ExactMass', 'Da'],
        ['TPSA', 'TPSA', 'Å²'],
        ['H-bond donors', 'HBondDonorCount'],
        ['H-bond acceptors', 'HBondAcceptorCount'],
        ['Rotatable bonds', 'RotatableBondCount'],
    ];

    function profileCard(tox, mode) {
        const compound = (tox && tox.compound) || {};
        const rows = PROFILE_ROWS.filter(([, key]) => compound[key] !== undefined && compound[key] !== null);
        let body;
        if (rows.length) {
            body = h('dl', { class: 'kv' }, rows.map(([label, key, unit, render]) =>
                h('div', {},
                    h('dt', { text: label }),
                    h('dd', {}, render ? render(compound[key]) : String(compound[key]),
                        unit ? h('span', { class: 'unit', text: unit }) : null))));
        } else {
            body = emptyState(mode === 'smiles'
                ? 'Molecular properties are retrieved from PubChem for chemical-name lookups only. Switch to “Chemical name” to see them.'
                : 'PubChem did not return computed properties for this compound.');
        }
        return card({ title: 'Molecular profile', iconName: 'hexagon', source: 'pubchem', span: 'span-7' }, body, rows.length ? pubchemFoot(compound) : null);
    }

    function sectionCard(opts, tox, mode, items, emptyMessage) {
        let body;
        if (items && items.length) body = textList(items);
        else body = emptyState(mode === 'smiles'
            ? 'PubChem information is only retrieved for chemical-name lookups. Switch to “Chemical name” to see it.'
            : emptyMessage);
        return card({ ...opts, source: 'pubchem' }, body, items && items.length ? pubchemFoot(tox && tox.compound) : null);
    }

    function ghsCard(tox, mode) {
        const all = (tox && tox.ghs_classification) || [];
        const codes = (tox && tox.hazard_codes) || [];
        const signals = [...new Set(all.filter((s) => /^(danger|warning)$/i.test(s.trim())).map((s) => s.trim()))];
        const notes = all.filter((s) => !/^(danger|warning)$/i.test(s.trim()) && !codes.includes(s));
        const parts = [];

        if (signals.length) {
            parts.push(h('div', {},
                h('div', { class: 'subhead', text: 'Signal word(s) reported' }),
                h('div', { class: 'signals' }, signals.map((word) =>
                    h('span', { class: `signal signal--${word.toLowerCase()}`, text: word.toUpperCase() })))));
        }
        if (codes.length) {
            parts.push(h('div', {},
                h('div', { class: 'subhead', text: 'Hazard statements' }),
                h('ul', { class: 'hcodes' }, codes.map((line) => {
                    const match = /^(H\d{3}[A-Za-z]*(?:\+H\d{3}[A-Za-z]*)*)\s*(.*)$/.exec(line);
                    return h('li', { class: 'hcode' },
                        h('span', { class: 'hcode__code', text: match ? match[1] : 'H' }),
                        h('span', { class: 'hcode__text', text: match ? (match[2] || line) : line }));
                }))));
        }
        if (notes.length) {
            parts.push(h('div', {}, h('div', { class: 'subhead', text: 'Aggregated notes' }), textList(notes)));
        }
        if (!parts.length) {
            parts.push(emptyState(mode === 'smiles'
                ? 'GHS data is only retrieved for chemical-name lookups. Switch to “Chemical name” to see it.'
                : 'No GHS classification was returned by PubChem for this compound.'));
        }
        if (signals.length > 1) {
            parts.push(h('p', { class: 'small muted', text: 'Different PubChem sources report different signal words for this compound; all are shown as reported.' }));
        }
        return card({ title: 'GHS classification', iconName: 'shield', source: 'pubchem', span: 'span-7' }, ...parts,
            all.length ? pubchemFoot(tox && tox.compound) : null);
    }

    function modelCard() {
        return card({ title: 'Model details', iconName: 'cpu', source: 'model', modifier: 'card--model', span: 'span-5' },
            h('dl', { class: 'kv' },
                h('div', {}, h('dt', { text: 'Assay' }), h('dd', { text: 'Tox21 SR-ATAD5' })),
                h('div', {}, h('dt', { text: 'Network' }), h('dd', { text: '2 × GCN layers → 2 dense layers → sigmoid' })),
                h('div', {}, h('dt', { text: 'Input' }), h('dd', { text: 'SMILES, up to 300 atoms (H, C, O, N, P, S, Cl, F encoded)' })),
                h('div', {}, h('dt', { text: 'Output' }), h('dd', { text: 'Single raw sigmoid value between 0 and 1' }))),
            h('div', {},
                h('div', { class: 'subhead', text: 'Limitations' }),
                h('ul', { class: 'limits' },
                    h('li', { text: 'Uncalibrated: training oversampled positives, so the output is not a probability.' }),
                    h('li', { text: 'Predicts one in-vitro assay only; says nothing about other toxicity endpoints.' }),
                    h('li', { text: 'No performance metrics are published for this model.' }),
                    h('li', { text: 'Other elements (e.g. Br, I, metals) are invisible to the encoder; the result depends on SMILES atom order.' }))));
    }

    /* ───────── Result rendering ───────── */
    function renderResults(data, mode) {
        const tox = data.toxicity_data || null;
        const compound = (tox && tox.compound) || {};
        const cidUrl = pubchemUrl(compound.cid);
        const smilesInput = mode === 'smiles';

        const head = h('section', { class: 'card result-head', 'aria-labelledby': 'result-title' },
            h('div', { class: 'result-head__top' },
                h('div', { class: 'result-head__title' },
                    h('div', { class: 'ident__label', text: smilesInput ? 'Direct SMILES input' : 'Query' }),
                    h('h2', { id: 'result-title', text: smilesInput ? 'Custom molecule' : data.chemical_name }),
                    compound.IUPACName ? h('p', { class: 'result-head__sub', text: `IUPAC name (PubChem): ${compound.IUPACName}` }) : null),
                h('div', { class: 'chips' },
                    h('span', { class: 'status', text: 'Analysis complete' }),
                    smilesInput ? h('span', { class: 'tag tag--model', text: 'No PubChem lookup' }) : h('span', { class: 'tag tag--pubchem', text: 'Resolved via PubChem' }))),
            h('div', { class: 'ident-grid' },
                copyField('SMILES' + (smilesInput ? '' : ' (from PubChem)'), data.smiles, data.smiles, 'SMILES'),
                cidUrl
                    ? copyField('PubChem CID', h('a', { href: cidUrl, target: '_blank', rel: 'noopener noreferrer', text: String(compound.cid) }), String(compound.cid), 'PubChem CID')
                    : null));

        const cards = h('div', { class: 'cards' },
            scoreCard(data),
            profileCard(tox, mode),
            sectionCard({ title: 'Toxicity information', iconName: 'skull', span: 'span-6', wide: true }, tox, mode,
                tox && tox.toxicity, 'PubChem did not return a toxicity summary for this compound.'),
            sectionCard({ title: 'Health hazards', iconName: 'heart-pulse', span: 'span-6', wide: true }, tox, mode,
                tox && tox.health_effects, 'PubChem did not return health hazard information for this compound.'),
            ghsCard(tox, mode),
            modelCard());

        const extra = [];
        if (tox && tox.safety_summary) {
            extra.push(card({ title: 'Hazards summary', iconName: 'alert', source: 'pubchem' },
                textList([tox.safety_summary]), pubchemFoot(compound)));
        }

        results.replaceChildren(head, cards, ...extra);
        results.hidden = false;
        results.focus({ preventScroll: true });
        results.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    }

    /* ───────── Feedback states ───────── */
    function showLoading(mode) {
        const message = mode === 'name'
            ? 'Resolving the name through PubChem and running the model…'
            : 'Running the model…';
        feedback.replaceChildren(
            h('div', { class: 'banner banner--loading', role: 'status' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), h('p', { text: message })),
            h('div', { class: 'skeleton-grid', 'aria-hidden': 'true' }, h('div', { class: 'skeleton' }), h('div', { class: 'skeleton' }), h('div', { class: 'skeleton' })));
    }

    function showError(message) {
        feedback.replaceChildren(h('div', { class: 'banner banner--error', role: 'alert' },
            icon('x-circle'),
            h('p', {}, h('strong', { text: 'Analysis failed. ' }), message)));
    }

    function setFieldError(message) {
        queryError.hidden = !message;
        clear(queryError);
        if (message) queryError.append(icon('alert'), message);
        queryInput.setAttribute('aria-invalid', message ? 'true' : 'false');
    }

    function setBusy(state) {
        busy = state;
        submitBtn.disabled = state;
        submitBtn.classList.toggle('is-loading', state);
        submitBtn.querySelector('.btn__label').textContent = state ? 'Analyzing…' : 'Analyze molecule';
        submitBtn.setAttribute('aria-busy', String(state));
        queryInput.readOnly = state;
        examplesBox.querySelectorAll('button').forEach((b) => { b.disabled = state; });
    }

    /* ───────── Modes and examples ───────── */
    function currentMode() { return form.elements.mode.value; }

    function renderExamples() {
        const mode = currentMode();
        const chips = mode === 'name'
            ? EXAMPLES.name.map((name) => h('button', { type: 'button', class: 'chip-btn', 'data-value': name, text: name }))
            : EXAMPLES.smiles.map(([smiles, label]) => h('button', { type: 'button', class: 'chip-btn', 'data-value': smiles, title: smiles, text: label }));
        clear(examplesBox);
        examplesBox.append(h('span', { class: 'examples__label', text: 'Try:' }), ...chips);
    }

    function applyMode() {
        const mode = currentMode();
        queryLabel.textContent = MODE_TEXT[mode].label;
        queryInput.placeholder = MODE_TEXT[mode].placeholder;
        queryInput.value = '';
        setFieldError('');
        renderExamples();
    }

    /* ───────── Request ───────── */
    async function analyze(mode, value) {
        const body = mode === 'smiles' ? { smiles: value } : { chemical_name: value };
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: controller.signal,
            });
            let data = null;
            try { data = await response.json(); } catch (_) { data = null; }
            if (response.ok && data && data.success) return data;
            throw new Error((data && data.error) || `The server returned an unexpected response (HTTP ${response.status}).`);
        } finally {
            clearTimeout(timer);
        }
    }

    async function onSubmit(event) {
        event.preventDefault();
        if (busy) return;
        const mode = currentMode();
        const value = queryInput.value.trim();
        if (!value) {
            setFieldError(mode === 'name' ? 'Please enter a chemical name.' : 'Please enter a SMILES string.');
            queryInput.focus();
            return;
        }
        setFieldError('');
        results.hidden = true;
        clear(results);
        showLoading(mode);
        setBusy(true);
        try {
            const data = await analyze(mode, value);
            clear(feedback);
            renderResults(data, mode);
        } catch (error) {
            results.hidden = true;
            if (error.name === 'AbortError') showError('The request timed out. PubChem may be slow or unreachable; please try again.');
            else if (error instanceof TypeError) showError('Could not reach the server. Check that the app is running and try again.');
            else showError(error.message);
        } finally {
            setBusy(false);
        }
    }

    form.addEventListener('submit', onSubmit);
    form.addEventListener('change', (event) => { if (event.target.name === 'mode') applyMode(); });
    queryInput.addEventListener('input', () => { if (queryError.hidden === false) setFieldError(''); });
    examplesBox.addEventListener('click', (event) => {
        const chip = event.target.closest('button[data-value]');
        if (!chip || busy) return;
        queryInput.value = chip.dataset.value;
        form.requestSubmit();
    });

    applyMode();
})();
