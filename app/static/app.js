const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const conversation = document.querySelector("#conversation");
const intro = document.querySelector("#intro");
const submitButton = form.querySelector("button[type='submit']");

document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
        questionInput.value = button.textContent.trim();
        resizeInput();
        questionInput.focus();
    });
});

questionInput.addEventListener("input", resizeInput);
questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();
    if (!question || submitButton.disabled) {
        return;
    }

    intro.classList.add("is-collapsed");
    appendMessage("user", question);
    questionInput.value = "";
    resizeInput();
    setLoading(true);
    const loadingMessage = appendLoadingMessage();
    scrollToLatest();

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "The request could not be completed.");
        }

        loadingMessage.remove();
        await appendAssistantMessage(data.answer, data.sources || []);
    } catch (error) {
        loadingMessage.remove();
        appendMessage(
            "assistant error-message",
            error.message || "Something went wrong. Please try again.",
        );
    } finally {
        setLoading(false);
        questionInput.focus();
        scrollToLatest();
    }
});

function appendMessage(role, text) {
    const message = document.createElement("article");
    message.className = `message ${role}`;

    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = role.startsWith("user") ? "You" : "Assistant";

    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = text;

    message.append(label, body);
    conversation.append(message);
    return message;
}

async function appendAssistantMessage(answer, sources) {
    const message = appendMessage("assistant", "");
    const body = message.querySelector(".message-body");
    await animateAnswer(body, answer);

    if (!sources.length) {
        return;
    }

    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    sourceList.setAttribute("aria-label", "Answer sources");

    sources.forEach((source, index) => {
        const card = document.createElement(source.source_url ? "a" : "div");
        card.className = "source-card";
        card.style.animationDelay = `${index * 70}ms`;

        if (source.source_url) {
            card.href = source.source_url;
            card.target = "_blank";
            card.rel = "noreferrer";
        }

        const provider = document.createElement("span");
        provider.className = "source-provider";
        provider.textContent = formatProvider(source.provider);

        const title = document.createElement("span");
        title.className = "source-title";
        title.textContent = source.source;

        card.append(provider, title);
        sourceList.append(card);
    });

    message.append(sourceList);
}

async function animateAnswer(element, answer) {
    const prefersReducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
    ).matches;

    if (prefersReducedMotion) {
        element.textContent = answer;
        return;
    }

    const parts = answer.split(/(\s+)/);
    let wordCount = 0;
    element.classList.add("is-typing");

    for (const part of parts) {
        if (!part) {
            continue;
        }

        if (/^\s+$/.test(part)) {
            element.append(document.createTextNode(part));
            continue;
        }

        const word = document.createElement("span");
        word.className = "answer-word";
        word.textContent = part;
        element.append(word);
        wordCount += 1;

        if (wordCount % 8 === 0) {
            scrollToLatest();
        }

        await wait(38);
    }

    element.classList.remove("is-typing");
}

function wait(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function appendLoadingMessage() {
    const message = document.createElement("article");
    message.className = "message assistant loading-message";

    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = "Searching knowledge";

    const body = document.createElement("div");
    body.className = "message-body";
    body.setAttribute("aria-label", "Loading answer");

    for (let index = 0; index < 3; index += 1) {
        const dot = document.createElement("span");
        dot.className = "loading-dot";
        body.append(dot);
    }

    message.append(label, body);
    conversation.append(message);
    return message;
}

function formatProvider(provider) {
    if (!provider || provider === "local") {
        return "Local document";
    }

    return provider.replaceAll("_", " ");
}

function resizeInput() {
    questionInput.style.height = "auto";
    questionInput.style.height = `${questionInput.scrollHeight}px`;
}

function setLoading(isLoading) {
    submitButton.disabled = isLoading;
    questionInput.disabled = isLoading;
}

function scrollToLatest() {
    window.setTimeout(() => {
        window.scrollTo({
            top: document.documentElement.scrollHeight,
            behavior: "smooth",
        });
    }, 80);
}
