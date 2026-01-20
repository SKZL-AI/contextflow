/**
 * Hook Registry for ContextFlow
 *
 * Central export point for all development hooks.
 * Boris Step 9: "PostToolUse Hook for formatting - Optional: Hooks-System for Code-Quality"
 *
 * Available Hooks:
 * - postEdit: Runs after file edits (formatting, linting)
 * - postWrite: Runs after new file creation (init files, headers)
 * - preCommit: Runs before git commits (tests, validation)
 */

const postEdit = require('./post-edit');
const postWrite = require('./post-write');
const preCommit = require('./pre-commit');

/**
 * Hook configuration and metadata
 */
const HOOKS_CONFIG = {
    version: '1.0.0',
    hooks: {
        postEdit: {
            description: 'Runs after file edits - auto-format and lint',
            triggers: ['Edit', 'Write'],
            enabled: true,
        },
        postWrite: {
            description: 'Runs after new file creation - ensure package structure',
            triggers: ['Write'],
            enabled: true,
        },
        preCommit: {
            description: 'Runs before git commits - tests and validation',
            triggers: ['git commit'],
            enabled: true,
        },
    },
};

/**
 * Execute a hook by name
 */
async function executeHook(hookName, context) {
    const hooks = { postEdit, postWrite, preCommit };

    if (!hooks[hookName]) {
        throw new Error(`Unknown hook: ${hookName}`);
    }

    const config = HOOKS_CONFIG.hooks[hookName];
    if (!config || !config.enabled) {
        return { success: true, skipped: true, reason: 'Hook disabled' };
    }

    return await hooks[hookName](context);
}

/**
 * List all available hooks
 */
function listHooks() {
    return Object.entries(HOOKS_CONFIG.hooks).map(([name, config]) => ({
        name,
        ...config,
    }));
}

module.exports = {
    // Individual hooks
    postEdit,
    postWrite,
    preCommit,

    // Utilities
    executeHook,
    listHooks,

    // Configuration
    config: HOOKS_CONFIG,
};
