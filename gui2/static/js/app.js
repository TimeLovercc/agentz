// State management
const state = {
    pipelines: {},
    selectedPipelineId: null,
    runs: {},
    runOrder: [],
    events: {},
    selectedRunId: null,
    runStream: null,
    pollingTimer: null,
    isFetchingRuns: false
};

// DOM references
const elements = {
    pipelineSelect: document.getElementById("pipeline-select"),
    pipelineDescription: document.getElementById("pipeline-description"),
    form: document.getElementById("pipeline-form"),
    dynamicFields: document.getElementById("dynamic-fields"),
    submitBtn: document.getElementById("submit-run"),
    runList: document.getElementById("run-list"),
    runDetail: document.getElementById("run-detail"),
    refreshRuns: document.getElementById("refresh-runs"),
    stopRun: document.getElementById("stop-run"),
    formFeedback: document.getElementById("form-feedback")
};

function init() {
    bindEventListeners();
    fetchPipelines();
    fetchRuns();
    startPollingRuns();
}

function bindEventListeners() {
    elements.pipelineSelect.addEventListener("change", handlePipelineChange);
    elements.form.addEventListener("submit", handleFormSubmit);
    elements.refreshRuns.addEventListener("click", () => fetchRuns(true));
    elements.stopRun.addEventListener("click", handleStopRun);
    window.addEventListener("beforeunload", cleanup);
}

function cleanup() {
    if (state.runStream) {
        state.runStream.close();
        state.runStream = null;
    }
    if (state.pollingTimer) {
        clearInterval(state.pollingTimer);
        state.pollingTimer = null;
    }
}

async function fetchPipelines() {
    try {
        const response = await fetch("/api/pipelines");
        if (!response.ok) throw new Error("Failed to load pipelines");

        const data = await response.json();
        (data.pipelines || []).forEach((pipeline) => {
            state.pipelines[pipeline.id] = pipeline;
        });

        renderPipelineOptions();
    } catch (error) {
        console.error(error);
        setFormFeedback("error", "Unable to load pipelines. Please refresh the page.");
    }
}

function renderPipelineOptions() {
    const select = elements.pipelineSelect;
    select.innerHTML = '<option value="">Select a pipeline…</option>';

    Object.values(state.pipelines).forEach((pipeline) => {
        const option = document.createElement("option");
        option.value = pipeline.id;
        option.textContent = pipeline.name;
        select.appendChild(option);
    });
}

function handlePipelineChange(event) {
    const pipelineId = event.target.value;
    state.selectedPipelineId = pipelineId || null;
    clearFormFeedback();

    if (!pipelineId) {
        elements.pipelineDescription.textContent = "";
        elements.dynamicFields.innerHTML = "";
        elements.submitBtn.disabled = true;
        return;
    }

    const pipeline = state.pipelines[pipelineId];
    if (!pipeline) {
        elements.pipelineDescription.textContent = "";
        elements.dynamicFields.innerHTML = "";
        elements.submitBtn.disabled = true;
        return;
    }

    elements.pipelineDescription.textContent = pipeline.description || "";
    renderPipelineFields(pipeline);
    elements.submitBtn.disabled = false;
}

function renderPipelineFields(pipeline) {
    elements.dynamicFields.innerHTML = "";

    (pipeline.fields || []).forEach((field) => {
        const wrapper = document.createElement("div");
        wrapper.className = "form-field";

        const label = document.createElement("label");
        label.setAttribute("for", `field-${field.id}`);
        label.textContent = field.required ? `${field.label} *` : field.label;

        const inputName = field.id;
        let input;
        if (field.type === "textarea") {
            input = document.createElement("textarea");
            input.rows = 6;
        } else {
            input = document.createElement("input");
            input.type = field.type || "text";
        }

        input.id = `field-${field.id}`;
        input.name = inputName;
        input.classList.add("input-control");
        if (field.placeholder) input.placeholder = field.placeholder;
        if (field.default) input.value = field.default;
        if (field.required) input.required = true;

        wrapper.appendChild(label);
        wrapper.appendChild(input);

        if (field.help_text) {
            const help = document.createElement("div");
            help.className = "form-help";
            help.textContent = field.help_text;
            wrapper.appendChild(help);
        }

        elements.dynamicFields.appendChild(wrapper);
    });
}

async function handleFormSubmit(event) {
    event.preventDefault();

    const pipelineId = state.selectedPipelineId;
    if (!pipelineId) {
        setFormFeedback("error", "Please select a pipeline before submitting.");
        return;
    }

    const pipeline = state.pipelines[pipelineId];
    if (!pipeline) {
        setFormFeedback("error", "Selected pipeline metadata not found.");
        return;
    }

    const formData = new FormData(elements.form);
    const inputs = {};

    (pipeline.fields || []).forEach((field) => {
        const rawValue = formData.get(field.id);
        const value = typeof rawValue === "string" ? rawValue.trim() : rawValue;
        inputs[field.id] = value ?? "";
    });

    elements.submitBtn.disabled = true;
    elements.submitBtn.textContent = "Submitting…";
    clearFormFeedback();

    try {
        const response = await fetch("/api/runs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                pipeline_id: pipelineId,
                inputs
            })
        });

        const data = await response.json();

        if (!response.ok) {
            const errorMessage =
                (data.details && data.details.join("\n")) ||
                data.error ||
                "Unable to create run.";
            setFormFeedback("error", errorMessage);
            return;
        }

        const run = data.run;
        if (run) {
            run.inputs = inputs;
            addOrUpdateRun(run);
            setFormFeedback("success", `${pipeline.name} submitted successfully.`);
            selectRun(run.id);
        }

        fetchRuns();
    } catch (error) {
        console.error(error);
        setFormFeedback("error", "Failed to submit run. Please try again.");
    } finally {
        elements.submitBtn.disabled = false;
        elements.submitBtn.textContent = "Submit Pipeline";
    }
}

function addOrUpdateRun(run) {
    const existing = state.runs[run.id] || {};
    state.runs[run.id] = { ...existing, ...run };

    if (!state.runOrder.includes(run.id)) {
        state.runOrder.push(run.id);
    }

    if (!state.events[run.id]) {
        state.events[run.id] = [];
    }

    renderRunList();
}

async function fetchRuns(force = false) {
    if (state.isFetchingRuns && !force) return;
    state.isFetchingRuns = true;

    try {
        const response = await fetch("/api/runs");
        if (!response.ok) throw new Error("Failed to fetch runs");

        const data = await response.json();
        (data.runs || []).forEach((run) => addOrUpdateRun(run));

        if (state.selectedRunId) {
            // Ensure detail view stays up-to-date
            const selectedRun = state.runs[state.selectedRunId];
            if (selectedRun) {
                renderRunDetail(selectedRun);
                updateStopButton(selectedRun);
            }
        }
    } catch (error) {
        console.error(error);
    } finally {
        state.isFetchingRuns = false;
    }
}

function renderRunList() {
    const container = elements.runList;
    container.innerHTML = "";

    const runs = state.runOrder
        .map((id) => state.runs[id])
        .filter(Boolean)
        .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

    if (runs.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <h3>No runs yet</h3>
                <p>Submit a pipeline to see it appear here.</p>
            </div>
        `;
        return;
    }

    runs.forEach((run) => {
        const card = document.createElement("div");
        card.className = "run-card";
        if (run.id === state.selectedRunId) {
            card.classList.add("selected");
        }
        card.dataset.runId = run.id;

        const header = document.createElement("div");
        header.className = "run-card-header";

        const title = document.createElement("h3");
        title.className = "run-card-title";
        title.textContent = run.pipeline_name || run.pipeline_id;

        const status = document.createElement("span");
        const statusClass = `status-badge status-${(run.status || "queued").replace(/\s+/g, "-")}`;
        status.className = statusClass;
        status.textContent = run.status || "queued";

        header.appendChild(title);
        header.appendChild(status);

        const meta = document.createElement("p");
        meta.className = "run-card-meta";
        const submitted = formatTimestamp(run.created_at);
        const duration = formatDuration(run.started_at, run.completed_at);
        meta.innerHTML = `<span>Submitted ${submitted}</span>${duration ? `<span>${duration}</span>` : ""}`;

        card.appendChild(header);
        card.appendChild(meta);

        card.addEventListener("click", () => selectRun(run.id));
        container.appendChild(card);
    });
}

function formatTimestamp(ts) {
    if (!ts) return "recently";
    const date = new Date(ts * 1000);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDuration(start, end) {
    if (!start || !end) return "";
    const diff = Math.max(0, end - start);
    const seconds = Math.round(diff);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return `${minutes}m ${remainder}s`;
}

async function selectRun(runId) {
    if (!runId) return;

    if (state.runStream) {
        state.runStream.close();
        state.runStream = null;
    }

    state.selectedRunId = runId;
    renderRunList();

    await fetchRunDetail(runId);
    const run = state.runs[runId];
    if (run) {
        connectToRunStream(runId);
        renderRunDetail(run);
        updateStopButton(run);
    }
}

async function fetchRunDetail(runId) {
    try {
        const response = await fetch(`/api/runs/${runId}`);
        if (!response.ok) throw new Error("Failed to fetch run detail");

        const run = await response.json();
        addOrUpdateRun(run);
    } catch (error) {
        console.error(error);
    }
}

function connectToRunStream(runId) {
    const source = new EventSource(`/api/runs/${runId}/stream`);

    source.onmessage = (event) => {
        if (!event.data) return;
        try {
            const data = JSON.parse(event.data);
            handleRunEvent(runId, data);
        } catch (error) {
            console.error("Error parsing stream data", error);
        }
    };

    source.onerror = () => {
        source.close();
    };

    state.runStream = source;
}

function handleRunEvent(runId, event) {
    const run = state.runs[runId] || {};
    run.status = event.run_status || run.status;
    state.runs[runId] = run;

    if (!state.events[runId]) {
        state.events[runId] = [];
    }

    if (event.event && event.event !== "stream_end") {
        const entry = normalizeEvent(event.event, event.payload);
        if (entry) {
            state.events[runId].push(entry);
            if (state.events[runId].length > 100) {
                state.events[runId].shift();
            }
        }
    }

    if (event.event === "summary") {
        if (event.payload) {
            run.result = event.payload.result || run.result;
            run.error = event.payload.error || run.error;
            run.status = event.payload.status || run.status;
        }
    }

    if (runId === state.selectedRunId) {
        renderRunDetail(run);
        updateStopButton(run);
    }

    renderRunList();
}

function normalizeEvent(type, payload = {}) {
    const timestamp = Date.now();

    switch (type) {
        case "status_update":
            return {
                timestamp,
                title: payload.title || "Status Update",
                message: payload.content || "",
                type
            };
        case "group_start":
            return {
                timestamp,
                title: payload.title || `Group ${payload.group_id || ""}`.trim(),
                message: "Group started",
                type
            };
        case "group_end":
            return {
                timestamp,
                title: payload.title || `Group ${payload.group_id || ""}`.trim(),
                message: payload.is_done ? "Group completed" : "Group ended",
                type
            };
        case "log_panel":
            return {
                timestamp,
                title: payload.title || "Log Panel",
                message: payload.content || "",
                type
            };
        case "error":
            return {
                timestamp,
                title: "Error",
                message: payload.message || "Pipeline error",
                type
            };
        case "cancelled":
            return {
                timestamp,
                title: "Cancelled",
                message: payload.message || "Run cancelled",
                type
            };
        case "summary":
            // handled separately
            return null;
        default:
            return {
                timestamp,
                title: type,
                message: JSON.stringify(payload, null, 2),
                type
            };
    }
}

function renderRunDetail(run) {
    const container = elements.runDetail;
    container.innerHTML = "";

    if (!run) {
        container.innerHTML = `
            <div class="empty-state">
                <h3>Select a run to inspect</h3>
                <p>Live execution logs and final reports will appear here.</p>
            </div>
        `;
        return;
    }

    const header = document.createElement("div");
    header.className = "detail-header";

    const title = document.createElement("h3");
    title.textContent = run.pipeline_name || run.pipeline_id;

    const status = document.createElement("span");
    status.className = `status-badge status-${(run.status || "queued").replace(/\s+/g, "-")}`;
    status.textContent = run.status || "queued";

    header.appendChild(title);
    header.appendChild(status);

    const detailStatus = document.createElement("div");
    detailStatus.className = "detail-status";
    const submittedAt = formatTimestamp(run.created_at);
    detailStatus.textContent = `Submitted ${submittedAt}`;

    const inputsSection = document.createElement("div");
    inputsSection.className = "detail-section";
    inputsSection.innerHTML = "<h4>Inputs</h4>";
    const inputsPre = document.createElement("pre");
    inputsPre.textContent = JSON.stringify(run.inputs || {}, null, 2);
    inputsSection.appendChild(inputsPre);

    const resultSection = document.createElement("div");
    resultSection.className = "detail-section";
    resultSection.innerHTML = "<h4>Result</h4>";
    const resultPre = document.createElement("pre");
    resultPre.textContent = run.result || "No result yet.";
    resultSection.appendChild(resultPre);

    const eventsSection = document.createElement("div");
    eventsSection.className = "detail-section";
    eventsSection.innerHTML = "<h4>Events</h4>";

    const eventsList = document.createElement("div");
    eventsList.className = "events-list";

    const events = state.events[run.id] || [];
    if (events.length === 0) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.innerHTML = "<h3>No events yet</h3><p>Streaming updates will appear here.</p>";
        eventsList.appendChild(empty);
    } else {
        events.forEach((entry) => {
            const item = document.createElement("div");
            item.className = "event-item";

            const heading = document.createElement("h5");
            heading.textContent = entry.title;

            const message = document.createElement("p");
            message.textContent = entry.message;

            item.appendChild(heading);
            item.appendChild(message);
            eventsList.appendChild(item);
        });
    }

    eventsSection.appendChild(eventsList);

    const fragments = [header, detailStatus, inputsSection, resultSection];

    if (run.error) {
        const errorSection = document.createElement("div");
        errorSection.className = "detail-section";
        errorSection.innerHTML = "<h4>Error</h4>";
        const errorPre = document.createElement("pre");
        errorPre.textContent = run.error;
        errorSection.appendChild(errorPre);
        fragments.push(errorSection);
    }

    fragments.push(eventsSection);

    fragments.forEach((node) => container.appendChild(node));
}

function handleStopRun() {
    const runId = state.selectedRunId;
    if (!runId) return;

    fetch(`/api/runs/${runId}/stop`, { method: "POST" })
        .then((response) => response.json())
        .then(() => {
            fetchRuns(true);
        })
        .catch((error) => console.error("Failed to stop run", error));
}

function updateStopButton(run) {
    if (!run) {
        elements.stopRun.disabled = true;
        return;
    }

    const stoppableStatuses = new Set(["queued", "running", "cancelling"]);
    elements.stopRun.disabled = !stoppableStatuses.has(run.status);
}

function startPollingRuns() {
    if (state.pollingTimer) return;
    state.pollingTimer = setInterval(() => fetchRuns(), 7000);
}

function setFormFeedback(type, message) {
    const node = elements.formFeedback;
    node.textContent = message;
    node.className = `form-feedback ${type} visible`;
}

function clearFormFeedback() {
    const node = elements.formFeedback;
    node.textContent = "";
    node.className = "form-feedback";
}

document.addEventListener("DOMContentLoaded", init);
