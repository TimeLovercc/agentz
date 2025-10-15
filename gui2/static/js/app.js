// State management
const state = {
    pipelines: {},
    selectedPipelineId: null,
    runs: {},
    runOrder: [],
    events: {},
    iterations: {},
    detailViewState: {},
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

    if (!state.iterations[run.id]) {
        state.iterations[run.id] = { order: [], map: {} };
    }

    if (!state.detailViewState[run.id]) {
        state.detailViewState[run.id] = { fullLogsExpanded: false };
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

    if (!state.iterations[runId]) {
        state.iterations[runId] = { order: [], map: {} };
    }

    if (event.event && event.event !== "stream_end") {
        updateIterationState(runId, event);
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
        finalizeIterationState(runId, run.status);
    }

    if (runId === state.selectedRunId) {
        renderRunDetail(run);
        updateStopButton(run);
    }

    renderRunList();
}

function normalizeEvent(type, payload = {}) {
    const timestamp = Date.now();
    const groupId = payload.group_id || null;

    switch (type) {
        case "status_update": {
            const statusTitle =
                payload.title ||
                formatAgentName(payload.item_id || "") ||
                (payload.content ? payload.content.split(":")[0] : "");
            return {
                timestamp,
                title: statusTitle || "Status Update",
                message: payload.content || "",
                type,
                groupId,
                raw: payload
            };
        }
        case "group_start":
            return {
                timestamp,
                title: payload.title || formatGroupTitle(payload.group_id, ""),
                message: "Group started",
                type,
                groupId,
                raw: payload
            };
        case "group_end":
            return {
                timestamp,
                title: payload.title || formatGroupTitle(payload.group_id, ""),
                message: payload.is_done ? "Group completed" : "Group ended",
                type,
                groupId,
                raw: payload
            };
        case "log_panel":
            return {
                timestamp,
                title: payload.title || "Log Panel",
                message: payload.content || "",
                type,
                groupId,
                raw: payload
            };
        case "error":
            return {
                timestamp,
                title: "Error",
                message: payload.message || "Pipeline error",
                type,
                groupId,
                raw: payload
            };
        case "cancelled":
            return {
                timestamp,
                title: "Cancelled",
                message: payload.message || "Run cancelled",
                type,
                groupId,
                raw: payload
            };
        case "summary":
            // handled separately
            return null;
        default:
            return {
                timestamp,
                title: type,
                message: JSON.stringify(payload, null, 2),
                type,
                groupId,
                raw: payload
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
    const duration = formatDuration(run.started_at, run.completed_at);
    detailStatus.textContent = duration
        ? `Submitted ${submittedAt} • Duration ${duration}`
        : `Submitted ${submittedAt}`;

    container.appendChild(header);
    container.appendChild(detailStatus);

    const eventsEntry = createDetailEntry("Events", buildEventsContent(run));
    container.appendChild(eventsEntry);

    const viewState = getDetailViewState(run.id);
    const fullLogsEntry = createDetailEntry(
        "Full Logs",
        buildFullLogsContent(run),
        {
            collapsible: true,
            defaultExpanded: !!viewState.fullLogsExpanded,
            onToggle: (expanded) => {
                viewState.fullLogsExpanded = expanded;
            }
        }
    );
    container.appendChild(fullLogsEntry);

    if (shouldShowResultEntry(run)) {
        const resultEntry = createDetailEntry(
            "Result",
            buildResultContent(run)
        );
        container.appendChild(resultEntry);
    }
}

function createDetailEntry(title, bodyNode, options = {}) {
    const { collapsible = false, defaultExpanded = true, description = "", onToggle = null } = options;

    const entry = document.createElement("section");
    entry.className = "detail-entry";

    if (collapsible) {
        entry.classList.add("collapsible");
    }

    if (!defaultExpanded) {
        entry.classList.add("collapsed");
    }

    const header = document.createElement(collapsible ? "button" : "div");
    header.className = "detail-entry-header";

    if (collapsible) {
        header.type = "button";
    }

    const titleSpan = document.createElement("span");
    titleSpan.className = "detail-entry-title";
    titleSpan.textContent = title;
    header.appendChild(titleSpan);

    if (description) {
        const descriptionSpan = document.createElement("span");
        descriptionSpan.className = "detail-entry-description";
        descriptionSpan.textContent = description;
        header.appendChild(descriptionSpan);
    }

    if (collapsible) {
        const icon = document.createElement("span");
        icon.className = "toggle-icon";
        icon.innerHTML = "▾";
        header.appendChild(icon);
    }

    const bodyWrapper = document.createElement("div");
    bodyWrapper.className = "detail-entry-body";
    bodyWrapper.appendChild(bodyNode);

    entry.appendChild(header);
    entry.appendChild(bodyWrapper);

    if (collapsible) {
        header.setAttribute("aria-expanded", defaultExpanded ? "true" : "false");
        header.addEventListener("click", () => {
            const isCollapsed = entry.classList.toggle("collapsed");
            header.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
            if (typeof onToggle === "function") {
                onToggle(!isCollapsed);
            }
        });
    }

    return entry;
}

function getDetailViewState(runId) {
    if (!state.detailViewState[runId]) {
        state.detailViewState[runId] = { fullLogsExpanded: false };
    }
    return state.detailViewState[runId];
}

function shouldShowResultEntry(run) {
    const status = (run.status || "").toLowerCase();
    return ["completed", "error", "cancelled"].includes(status);
}

function buildEventsContent(run) {
    const wrapper = document.createElement("div");
    wrapper.className = "events-summary";

    const store = state.iterations[run.id];
    const activeIds = store ? store.order.filter((groupId) => store.map[groupId]) : [];

    if (!store || activeIds.length === 0) {
        const placeholder = document.createElement("p");
        placeholder.className = "events-empty";
        const status = (run.status || "").toLowerCase();

        if (status === "running" || status === "queued" || status === "cancelling") {
            placeholder.textContent = "Waiting for active tasks…";
        } else if (store && store.terminalStatus === "error") {
            placeholder.textContent = "Pipeline encountered an error.";
        } else if (store && store.terminalStatus === "cancelled") {
            placeholder.textContent = "Pipeline cancelled.";
        } else if (status === "completed") {
            placeholder.textContent = "Pipeline completed.";
        } else {
            placeholder.textContent = "No active tasks.";
        }

        wrapper.appendChild(placeholder);
        return wrapper;
    }

    activeIds.forEach((groupId) => {
        const meta = store.map[groupId];
        if (!meta) return;

        const item = document.createElement("div");
        item.className = "event-summary-item";

        const summaryHeader = document.createElement("div");
        summaryHeader.className = "event-summary-header";

        const title = document.createElement("span");
        title.className = "event-iteration-title";
        title.textContent = meta.title || formatGroupTitle(groupId);

        const badge = document.createElement("span");
        badge.className = `status-badge status-${(meta.status || "pending").replace(/\s+/g, "-")}`;
        badge.textContent = formatStatusLabel(meta.status);

        summaryHeader.appendChild(title);
        summaryHeader.appendChild(badge);

        const agentLine = document.createElement("p");
        agentLine.className = "event-agent";
        if (meta.currentAgent) {
            agentLine.textContent = `Working agent: ${meta.currentAgent}`;
        } else if (meta.lastAgent) {
            agentLine.textContent = `Last agent: ${meta.lastAgent}`;
        } else {
            agentLine.textContent = "Waiting for activity";
        }

        item.appendChild(summaryHeader);
        item.appendChild(agentLine);

        wrapper.appendChild(item);
    });

    return wrapper;
}

function buildFullLogsContent(run) {
    const wrapper = document.createElement("div");
    wrapper.className = "full-logs";

    const inputsPre = createPre(JSON.stringify(run.inputs || {}, null, 2));
    wrapper.appendChild(createDetailSection("Inputs", inputsPre));

    const resultPre = createPre(run.result || "No result yet.");
    wrapper.appendChild(createDetailSection("Result Snapshot", resultPre));

    if (run.error) {
        wrapper.appendChild(createDetailSection("Error", createPre(run.error)));
    }

    const eventsSection = document.createElement("div");
    eventsSection.className = "detail-section";
    eventsSection.innerHTML = "<h4>Event Stream</h4>";

    const eventsList = document.createElement("div");
    eventsList.className = "events-list logs";

    const events = state.events[run.id] || [];
    if (events.length === 0) {
        const placeholder = document.createElement("p");
        placeholder.className = "events-empty";
        placeholder.textContent = "No events recorded yet.";
        eventsList.appendChild(placeholder);
    } else {
        events.forEach((entry) => {
            const item = document.createElement("div");
            item.className = "event-item";

            const heading = document.createElement("div");
            heading.className = "event-item-header";

            const title = document.createElement("h5");
            title.textContent = entry.title;
            heading.appendChild(title);

            const time = document.createElement("span");
            time.className = "event-time";
            time.textContent = formatEventTime(entry.timestamp);
            heading.appendChild(time);

            const message = document.createElement("p");
            message.textContent = entry.message;

            item.appendChild(heading);
            item.appendChild(message);
            eventsList.appendChild(item);
        });
    }

    eventsSection.appendChild(eventsList);
    wrapper.appendChild(eventsSection);

    return wrapper;
}

function buildResultContent(run) {
    const wrapper = document.createElement("div");
    wrapper.className = "result-wrapper";

    if (run.completed_at) {
        const meta = document.createElement("p");
        meta.className = "result-meta";
        meta.textContent = `Completed ${formatTimestamp(run.completed_at)}`;
        wrapper.appendChild(meta);
    }

    wrapper.appendChild(
        createDetailSection("Final Output", createPre(run.result || "No result captured."))
    );

    if (run.error) {
        wrapper.appendChild(createDetailSection("Error Details", createPre(run.error)));
    }

    return wrapper;
}

function createDetailSection(title, contentNode) {
    const section = document.createElement("div");
    section.className = "detail-section";
    const heading = document.createElement("h4");
    heading.textContent = title;
    section.appendChild(heading);
    section.appendChild(contentNode);
    return section;
}

function createPre(content) {
    const pre = document.createElement("pre");
    pre.textContent = content;
    return pre;
}

function formatEventTime(timestamp) {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function ensureIterationStore(runId) {
    if (!state.iterations[runId]) {
        state.iterations[runId] = { order: [], map: {}, terminalStatus: null };
    }
    return state.iterations[runId];
}

function removeIterationEntry(store, groupId) {
    if (!store || !groupId) return;
    if (store.map[groupId]) {
        delete store.map[groupId];
        store.order = store.order.filter((id) => id !== groupId);
    }
}

function updateIterationState(runId, event) {
    if (!event || !event.event) return;
    const payload = event.payload || {};
    const groupId = payload.group_id;

    if (!groupId) {
        if (event.event === "cancelled" || event.event === "error") {
            finalizeIterationState(runId, event.event);
        }
        return;
    }

    const store = ensureIterationStore(runId);

    if (!store.map[groupId]) {
        store.map[groupId] = {
            title: formatGroupTitle(groupId, payload.title),
            status: "pending",
            currentAgent: null,
            lastAgent: null,
            lastMessage: "",
            updatedAt: Date.now()
        };
        store.order.push(groupId);
    }

    const meta = store.map[groupId];
    meta.updatedAt = Date.now();

    if (payload.title) {
        meta.title = payload.title;
    }

    switch (event.event) {
        case "group_start":
            meta.status = "running";
            meta.currentAgent = null;
            break;
        case "group_end":
            meta.status = "completed";
            meta.currentAgent = null;
            removeIterationEntry(store, groupId);
            break;
        case "status_update": {
            const activity = parseActivityFromPayload(payload);
            meta.lastMessage = activity.message || meta.lastMessage;

            if (activity.completed) {
                meta.status = "completed";
                meta.lastAgent = activity.name || meta.lastAgent;
                meta.currentAgent = null;
                removeIterationEntry(store, groupId);
            } else if (activity.waiting) {
                meta.currentAgent = activity.message || activity.name || meta.currentAgent;
                meta.status = "running";
            } else if (activity.isActive && activity.name) {
                meta.currentAgent = activity.name;
                meta.status = "running";
            } else if (activity.name) {
                meta.lastAgent = activity.name;
            }
            break;
        }
        case "error":
            meta.status = "error";
            meta.currentAgent = null;
            removeIterationEntry(store, groupId);
            break;
        case "cancelled":
            meta.status = "cancelled";
            meta.currentAgent = null;
            removeIterationEntry(store, groupId);
            break;
        default:
            break;
    }
}

function finalizeIterationState(runId, runStatus) {
    const store = state.iterations[runId];
    if (!store) return;

    const status = (runStatus || "").toLowerCase();
    const terminal =
        status === "error" ? "error" : status === "cancelled" ? "cancelled" : "completed";

    store.order = [];
    store.map = {};
    store.terminalStatus = terminal;
}

function parseActivityFromPayload(payload) {
    const message = (payload.content || "").trim();
    const title = payload.title || "";
    const itemId = payload.item_id || "";

    let candidate = title || itemId;
    if (!candidate && message) {
        candidate = message.includes(":") ? message.split(":")[0] : message;
    }

    const formattedName = formatAgentName(candidate);
    const lowerMessage = message.toLowerCase();
    const waiting = lowerMessage.includes("waiting");
    const completed = lowerMessage.includes("completed") || Boolean(payload.is_done);
    const isActive = !waiting && !completed;

    return {
        name: formattedName,
        message: message || candidate || "",
        waiting,
        isActive,
        completed
    };
}

function formatAgentName(raw) {
    if (!raw) return "";

    let text = raw.replace(/^iter[:\-]\d+:/i, "");
    if (text.includes(":")) {
        const parts = text.split(":").map((part) => part.trim()).filter(Boolean);
        if (parts.length > 1) {
            text = parts.slice(1).join(" ");
        } else {
            text = parts[0];
        }
    }

    text = text
        .replace(/\btool\b/gi, "")
        .replace(/[_-]/g, " ")
        .replace(/\s+/g, " ")
        .trim();

    if (!text) return "";

    return text
        .split(" ")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
}

function formatGroupTitle(groupId, fallbackTitle) {
    if (fallbackTitle && fallbackTitle.trim()) {
        return fallbackTitle;
    }
    if (!groupId) {
        return "Iteration";
    }
    if (groupId === "iter-final") {
        return "Final Summary";
    }
    const match = groupId.match(/iter[-:](\d+)/);
    if (match) {
        const index = parseInt(match[1], 10);
        return `Iteration ${index + 1}`;
    }
    return formatAgentName(groupId);
}

function formatStatusLabel(status) {
    switch ((status || "pending").toLowerCase()) {
        case "running":
            return "Running";
        case "completed":
            return "Completed";
        case "error":
            return "Error";
        case "cancelled":
            return "Cancelled";
        case "cancelling":
            return "Cancelling";
        default:
            return "Pending";
    }
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
