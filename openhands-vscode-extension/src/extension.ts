import * as vscode from 'vscode';
import axios from 'axios';

// Define the structure of the response for the contextual chat
interface ContextualChatResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    provider: string;
    choices: Array<{
        index: number;
        message?: {
            role: string;
            content: string;
        };
        delta?: {
            role?: string;
            content?: string;
        };
        finish_reason: string | null;
    }>;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
}

// Define the structure of the response for the code agent
interface CodeAgentResponse {
    generated_code: string;
    explanation?: string;
    request_params?: any; // Or a more specific type if defined
    model_used?: string;
}

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

        if (!userPrompt) {
            // vscode.window.showInformationMessage('No prompt provided.'); // Optional: too noisy?
            return;
        }

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

                const requestBody = {
                    prompt: userPrompt,
                    selected_text: selectedText,
                    active_file_content: activeFileContent,
                    model: "auto",
                    provider: null,
                    model_quality: "balanced",
                    stream: false
                };

                progress.report({ increment: 25, message: "Sending request..." });

                const response = await axios.post<ContextualChatResponse>(apiUrl, requestBody, {
                    headers: { 'Content-Type': 'application/json' }
                });

                progress.report({ increment: 75, message: "Processing response..." });

                if (response.data && response.data.choices && response.data.choices.length > 0) {
                    const firstChoice = response.data.choices[0];
                    let contentToDisplay = "No message content found.";

                    if (firstChoice.message && firstChoice.message.content) {
                        contentToDisplay = firstChoice.message.content;
                    } else if (firstChoice.delta && firstChoice.delta.content) {
                        contentToDisplay = firstChoice.delta.content;
                    }

                    const newDocument = await vscode.workspace.openTextDocument({
                        content: contentToDisplay,
                        language: 'markdown'
                    });
                    await vscode.window.showTextDocument(newDocument, vscode.ViewColumn.Beside);
                    // vscode.window.showInformationMessage('OpenHands response received.'); // Optional
                } else {
                    vscode.window.showErrorMessage('Received an empty or invalid response from OpenHands.');
                }
                progress.report({ increment: 100 });
            });

        } catch (error: any) {
            console.error("Error calling OpenHands API:", error);
            let errorMessage = "Failed to get response from OpenHands.";
            if (axios.isAxiosError(error)) {
                if (error.response) {
                    errorMessage += ` Server responded with ${error.response.status}: ${JSON.stringify(error.response.data, null, 2)}`;
                } else if (error.request) {
                    errorMessage += " No response received from the server. Is it running and accessible?";
                } else {
                    errorMessage += ` Error: ${error.message}`;
                }
            } else {
                errorMessage += ` Error: ${error.message}`;
            }
            vscode.window.showErrorMessage(errorMessage);
        }
    });
    context.subscriptions.push(askOpenHandsDisposable);

    // Command 2: Ask OpenHands Code Agent
    let codeAgentDisposable = vscode.commands.registerCommand('openhands.invokeCodeAgent', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor found.');
            return;
        }

        const selectedText = editor.document.getText(editor.selection); // This is our context

        const userInstruction = await vscode.window.showInputBox({
            prompt: "Enter your coding instruction for OpenHands Code Agent:",
            placeHolder: "e.g., Create a Python function for X, refactor this selected code..."
        });

        if (!userInstruction) {
            // vscode.window.showInformationMessage('No instruction provided for Code Agent.'); // Optional
            return;
        }

        let language: string | undefined = undefined;
        if (editor.document.languageId && editor.document.languageId !== 'plaintext') { // Don't send plaintext as language
            language = editor.document.languageId;
        }

        const config = vscode.workspace.getConfiguration('openhands.api');
        const baseUrl = config.get<string>('baseUrl', 'http://localhost:8000');
        const apiUrl = `${baseUrl}/v1/agents/code/invoke`;

        try {
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "OpenHands Code Agent is working...",
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: "Preparing request..." });

                const requestBody = {
                    instruction: userInstruction,
                    context: selectedText || null,
                    language: language,
                    model_quality: "best_quality", // Default to best for code agent
                    provider: null
                };

                progress.report({ increment: 25, message: "Sending request to Code Agent..." });

                const response = await axios.post<CodeAgentResponse>(apiUrl, requestBody, {
                    headers: { 'Content-Type': 'application/json' }
                });

                progress.report({ increment: 75, message: "Processing response..." });

                const { generated_code, explanation, model_used } = response.data;

                if (generated_code) {
                    await editor.edit(editBuilder => {
                        if (editor.selection.isEmpty) {
                            editBuilder.insert(editor.selection.active, generated_code);
                        } else {
                            editBuilder.replace(editor.selection, generated_code);
                        }
                    });
                    // vscode.window.showInformationMessage(`Code inserted. Model used: ${model_used || 'N/A'}`); // Optional
                } else if (!explanation) {
                     vscode.window.showWarningMessage('Code Agent returned no code and no explanation.');
                }

                if (explanation) {
                    const explanationContent = `OpenHands Code Agent Explanation (Model: ${model_used || 'N/A'}):\n\n${explanation}`;
                    const explanationDoc = await vscode.workspace.openTextDocument({
                        content: explanationContent,
                        language: 'markdown'
                    });
                    await vscode.window.showTextDocument(explanationDoc, { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true });
                }
                progress.report({ increment: 100 });
            });

        } catch (error: any) {
            console.error("Error calling OpenHands Code Agent API:", error);
            let errorMessage = "Failed to get response from OpenHands Code Agent.";
            if (axios.isAxiosError(error)) {
                if (error.response) {
                    errorMessage += ` Server responded with ${error.response.status}: ${JSON.stringify(error.response.data, null, 2)}`;
                } else if (error.request) {
                    errorMessage += " No response received from the server.";
                } else {
                    errorMessage += ` Error: ${error.message}`;
                }
            } else {
                errorMessage += ` Error: ${error.message}`;
            }
            vscode.window.showErrorMessage(errorMessage);
        }
    });
    context.subscriptions.push(codeAgentDisposable);
}

export function deactivate() {}
