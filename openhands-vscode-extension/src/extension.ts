import * as vscode from 'vscode';
import axios from 'axios';

// --- Interface Definitions (from backend models) ---

interface ContextualChatResponse {
    // Assuming relevant fields based on previous definitions
    choices: Array<{ message?: { content: string }; delta?: { content?: string } }>;
    // ... other fields if needed by the extension
}

interface HumanInputRequest {
    tool_call_id: string;
    question_for_human: string;
}

interface CodeAgentResponse {
    generated_code?: string | null;
    explanation?: string | null;
    agent_status: string; // "completed", "requires_human_input", "error"
    human_input_request?: HumanInputRequest | null;
    model_used?: string | null;
    error_details?: string | null;
    // request_params?: any; // Not directly used by VSCode extension display typically
}

interface PlanStep {
    step_id: string;
    description: string;
    status: string;
    agent_to_use?: string | null;
    tool_to_use?: string | null;
    result?: string | null;
    error_details?: string | null;
}

interface Plan {
    plan_id: string;
    goal: string;
    steps: PlanStep[];
    plan_status: string;
    final_output?: string | null;
    error_details?: string | null;
}

interface PlanningAgentResponse {
    plan?: Plan | null;
    agent_status: string; // "plan_generated", "error", "paused_for_human_input" (if planner itself pauses)
    error_details?: string | null;
    thread_id?: string | null;
    human_input_request?: HumanInputRequest | null; // For HITL relayed from sub-agents
}


// --- Helper Functions ---

function formatPlanForDisplay(plan: Plan): string {
    let content = `# Plan Result\n\n`;
    content += `## Goal:\n${plan.goal}\n\n`;
    content += `## Overall Status:\n${plan.plan_status}\n\n`;

    if (plan.final_output) {
        content += `## Final Output/Summary:\n${plan.final_output}\n\n`;
    }
    if (plan.error_details) {
        content += `## Plan Error Details:\n${plan.error_details}\n\n`;
    }

    content += `## Steps:\n`;
    if (plan.steps && plan.steps.length > 0) {
        plan.steps.forEach((step, index) => {
            content += `${index + 1}. **[${step.status || 'pending'}]** ${step.description}\n`;
            if (step.agent_to_use) content += `   - Agent to use: \`${step.agent_to_use}\`\n`;
            if (step.tool_to_use) content += `   - Tool to use: \`${step.tool_to_use}\`\n`;
            if (step.result) content += `   - Result: ${step.result.substring(0, 200)}${step.result.length > 200 ? '...' : ''}\n`; // Show snippet of result
            if (step.error_details) content += `   - Error: ${step.error_details}\n`;
            content += "\n";
        });
    } else {
        content += "No steps were generated for this plan.\n";
    }
    return content;
}


// --- Extension Activation ---

export function activate(context: vscode.ExtensionContext) {
    console.log('Congratulations, your extension "openhands-ai-assistant" is now active!');

    // Command 1: Ask OpenHands (Contextual Chat)
    let askOpenHandsDisposable = vscode.commands.registerCommand('openhands.askOpenHands', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor found.');
            return;
        }
        const selectedText = editor.document.getText(editor.selection);
        const activeFileContent = editor.document.getText();
        const userPrompt = await vscode.window.showInputBox({
            prompt: "What would you like to ask OpenHands?",
            placeHolder: "e.g., Explain this code, suggest improvements, write a test case..."
        });
        if (!userPrompt) return;

        const config = vscode.workspace.getConfiguration('openhands.api');
        const baseUrl = config.get<string>('baseUrl', 'http://localhost:8000');
        const apiUrl = `${baseUrl}/v1/contextual_chat/completions`;

        try {
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Asking OpenHands...",
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: "Preparing request..." });
                const requestBody = { prompt: userPrompt, selected_text: selectedText, active_file_content: activeFileContent, model: "auto", provider: null, model_quality: "balanced", stream: false };
                progress.report({ increment: 25, message: "Sending request..." });
                const response = await axios.post<ContextualChatResponse>(apiUrl, requestBody, { headers: { 'Content-Type': 'application/json' }});
                progress.report({ increment: 75, message: "Processing response..." });

                if (response.data && response.data.choices && response.data.choices.length > 0) {
                    const firstChoice = response.data.choices[0];
                    let contentToDisplay = (firstChoice.message?.content || firstChoice.delta?.content || "No message content found.");
                    const newDocument = await vscode.workspace.openTextDocument({ content: contentToDisplay, language: 'markdown' });
                    await vscode.window.showTextDocument(newDocument, vscode.ViewColumn.Beside);
                } else { vscode.window.showErrorMessage('Received an empty or invalid response from OpenHands.'); }
                progress.report({ increment: 100 });
            });
        } catch (error: any) {
            console.error("Error calling OpenHands API:", error);
            let errorMessage = "Failed to get response from OpenHands.";
            if (axios.isAxiosError(error)) {
                if (error.response) errorMessage += ` Server responded with ${error.response.status}: ${JSON.stringify(error.response.data, null, 2)}`;
                else if (error.request) errorMessage += " No response received from server.";
                else errorMessage += ` Error: ${error.message}`;
            } else { errorMessage += ` Error: ${error.message}`; }
            vscode.window.showErrorMessage(errorMessage);
        }
    });
    context.subscriptions.push(askOpenHandsDisposable);

    // Command 2: Ask OpenHands Code Agent
    let codeAgentDisposable = vscode.commands.registerCommand('openhands.invokeCodeAgent', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) { vscode.window.showErrorMessage('No active text editor found.'); return; }
        const selectedText = editor.document.getText(editor.selection);
        const userInstruction = await vscode.window.showInputBox({ prompt: "Enter coding instruction for Code Agent:", placeHolder: "e.g., Create a Python function..." });
        if (!userInstruction) return;

        let language = (editor.document.languageId && editor.document.languageId !== 'plaintext') ? editor.document.languageId : undefined;
        let projectDirectory: string | undefined;
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders && workspaceFolders.length > 0) {
            projectDirectory = workspaceFolders[0].uri.fsPath;
            vscode.window.showInformationMessage(`OpenHands Code Agent using project context: ${projectDirectory}`);
        } else {
            vscode.window.showWarningMessage("No workspace folder open. File system tools will be unavailable for Code Agent.");
        }

        const config = vscode.workspace.getConfiguration('openhands.api');
        const baseUrl = config.get<string>('baseUrl', 'http://localhost:8000');
        const apiUrl = `${baseUrl}/v1/agents/code/invoke`;

        try {
            await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "OpenHands Code Agent is working...", cancellable: false }, async (progress) => {
                progress.report({ increment: 0, message: "Preparing request..." });
                const requestBody = { instruction: userInstruction, context: selectedText || null, language, project_directory: projectDirectory, model_quality: "best_quality", provider: null };
                progress.report({ increment: 25, message: "Sending request..." });
                const response = await axios.post<CodeAgentResponse>(apiUrl, requestBody, { headers: { 'Content-Type': 'application/json' }});
                progress.report({ increment: 75, message: "Processing response..." });

                const { generated_code, explanation, model_used, agent_status, human_input_request, error_details } = response.data;

                if (agent_status === "requires_human_input" && human_input_request) {
                    vscode.window.showInformationMessage(`Code Agent paused for your input: ${human_input_request.question_for_human}. Use thread ID: ${ (response.data as any).thread_id || 'N/A'} and Tool Call ID: ${human_input_request.tool_call_id} to resume via API.`);
                } else if (agent_status === "error") {
                    vscode.window.showErrorMessage(`Code Agent Error: ${error_details || 'Unknown error'}`);
                }

                if (generated_code) {
                    await editor.edit(editBuilder => {
                        if (editor.selection.isEmpty) editBuilder.insert(editor.selection.active, generated_code);
                        else editBuilder.replace(editor.selection, generated_code);
                    });
                }
                if (explanation) {
                    const explanationContent = `OpenHands Code Agent Explanation (Model: ${model_used || 'N/A'} | Status: ${agent_status}):\n\n${explanation}`;
                    const explanationDoc = await vscode.workspace.openTextDocument({ content: explanationContent, language: 'markdown' });
                    await vscode.window.showTextDocument(explanationDoc, { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true });
                }
                if (!generated_code && !explanation && agent_status === "completed") {
                    vscode.window.showInformationMessage('Code Agent completed without generating new code or explanation.');
                }
                progress.report({ increment: 100 });
            });
        } catch (error: any) {
            console.error("Error calling Code Agent API:", error);
            let errorMessage = "Failed to get response from Code Agent.";
            if (axios.isAxiosError(error)) {
                if (error.response) errorMessage += ` Server responded with ${error.response.status}: ${JSON.stringify(error.response.data, null, 2)}`;
                else if (error.request) errorMessage += " No response received from server.";
                else errorMessage += ` Error: ${error.message}`;
            } else { errorMessage += ` Error: ${error.message}`; }
            vscode.window.showErrorMessage(errorMessage);
        }
    });
    context.subscriptions.push(codeAgentDisposable);

    // Command 3: Invoke Planning Agent
    let planningAgentDisposable = vscode.commands.registerCommand('openhands.invokePlanningAgent', async () => {
        const userGoal = await vscode.window.showInputBox({
            prompt: "Enter your high-level goal for the Planning Agent",
            placeHolder: "e.g., Implement a new feature X, Refactor module Y for better performance..."
        });
        if (!userGoal) return;

        let projectDirectory: string | undefined;
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders && workspaceFolders.length > 0) {
            projectDirectory = workspaceFolders[0].uri.fsPath;
            vscode.window.showInformationMessage(`Planning Agent using project context: ${projectDirectory}`);
        } else {
            vscode.window.showWarningMessage("No workspace folder open. Planning Agent may have limited file system context.");
        }

        const config = vscode.workspace.getConfiguration('openhands.api');
        const baseUrl = config.get<string>('baseUrl', 'http://localhost:8000');
        const apiUrl = `${baseUrl}/v1/agents/plan/invoke`;

        try {
            await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "OpenHands Planning Agent is working...", cancellable: false }, async (progress) => {
                progress.report({ increment: 0, message: "Preparing request..."});
                const requestBody = { goal: userGoal, project_directory: projectDirectory }; // thread_id is managed by backend for new plans
                progress.report({ increment: 25, message: "Sending request..." });
                const response = await axios.post<PlanningAgentResponse>(apiUrl, requestBody, { headers: { 'Content-Type': 'application/json' }});
                progress.report({ increment: 75, message: "Processing response..."});

                const { plan, agent_status, error_details, thread_id, human_input_request } = response.data;

                if (agent_status === "requires_human_input" && human_input_request) {
                    vscode.window.showInformationMessage(`Planning Agent paused for your input: ${human_input_request.question_for_human}. Use Thread ID: ${thread_id || 'N/A'} and Tool Call ID: ${human_input_request.tool_call_id} to resume via the /v1/agents/resume API endpoint.`);
                } else if (agent_status === "error" || !plan) {
                    vscode.window.showErrorMessage(`Planning Agent Error: ${error_details || agent_status || 'Failed to generate or retrieve plan.'}`);
                } else if (plan) {
                    const planContent = formatPlanForDisplay(plan);
                    const document = await vscode.workspace.openTextDocument({ content: planContent, language: 'markdown' });
                    await vscode.window.showTextDocument(document, vscode.ViewColumn.Beside);
                    vscode.window.showInformationMessage(`Planning Agent finished: ${agent_status}. Thread ID: ${thread_id}`);
                }
                progress.report({ increment: 100 });
            });
        } catch (error: any) {
            console.error("Error calling Planning Agent API:", error);
            let errorMessage = "Failed to get response from Planning Agent.";
            if (axios.isAxiosError(error)) {
                if (error.response) errorMessage += ` Server responded with ${error.response.status}: ${JSON.stringify(error.response.data, null, 2)}`;
                else if (error.request) errorMessage += " No response received from server.";
                else errorMessage += ` Error: ${error.message}`;
            } else { errorMessage += ` Error: ${error.message}`; }
            vscode.window.showErrorMessage(errorMessage);
        }
    });
    context.subscriptions.push(planningAgentDisposable);
}

export function deactivate() {}
```
