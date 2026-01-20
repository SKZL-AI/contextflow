/**
 * Post-Edit Hook for ContextFlow
 *
 * Runs after file edits to ensure code quality:
 * - Auto-format Python with black
 * - Run ruff for linting
 * - Check type hints with mypy (optional)
 *
 * Boris Step 9: "PostToolUse Hook for formatting"
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const path = require('path');

const execAsync = promisify(exec);

// Configuration
const CONFIG = {
    formatters: {
        python: {
            black: true,
            ruff: true,
            mypy: false, // Optional - enable if needed
        },
    },
    timeout: 30000, // 30 seconds
    verbose: process.env.HOOK_VERBOSE === 'true',
};

/**
 * Logger utility for debugging
 */
const logger = {
    info: (msg, data = {}) => {
        if (CONFIG.verbose) {
            console.log(`[post-edit] INFO: ${msg}`, JSON.stringify(data));
        }
    },
    error: (msg, data = {}) => {
        console.error(`[post-edit] ERROR: ${msg}`, JSON.stringify(data));
    },
    success: (msg, data = {}) => {
        if (CONFIG.verbose) {
            console.log(`[post-edit] SUCCESS: ${msg}`, JSON.stringify(data));
        }
    },
};

/**
 * Run a command with timeout and error handling
 */
async function runCommand(command, options = {}) {
    const { timeout = CONFIG.timeout, ignoreErrors = false } = options;

    try {
        const { stdout, stderr } = await execAsync(command, { timeout });
        return { success: true, stdout, stderr };
    } catch (error) {
        if (ignoreErrors) {
            logger.info(`Command failed (ignored): ${command}`, { error: error.message });
            return { success: false, error: error.message };
        }
        throw error;
    }
}

/**
 * Format Python file with black
 */
async function formatWithBlack(filePath) {
    logger.info('Running black formatter', { filePath });
    const result = await runCommand(`black "${filePath}" --quiet`, { ignoreErrors: true });
    if (result.success) {
        logger.success('Black formatting complete', { filePath });
    }
    return result;
}

/**
 * Lint Python file with ruff and auto-fix
 */
async function lintWithRuff(filePath) {
    logger.info('Running ruff linter', { filePath });
    const result = await runCommand(`ruff check "${filePath}" --fix --quiet`, { ignoreErrors: true });
    if (result.success) {
        logger.success('Ruff linting complete', { filePath });
    }
    return result;
}

/**
 * Type check Python file with mypy (optional)
 */
async function typeCheckWithMypy(filePath) {
    logger.info('Running mypy type checker', { filePath });
    const result = await runCommand(`mypy "${filePath}" --ignore-missing-imports`, { ignoreErrors: true });
    if (result.success) {
        logger.success('Mypy type check complete', { filePath });
    }
    return result;
}

/**
 * Main post-edit hook function
 */
module.exports = async function postEdit(context) {
    const { filePath, toolName } = context;
    const results = [];

    logger.info('Post-edit hook triggered', { filePath, toolName });

    // Handle Python files
    if (filePath && filePath.endsWith('.py')) {
        const { python } = CONFIG.formatters;

        // Format with black
        if (python.black) {
            results.push(await formatWithBlack(filePath));
        }

        // Lint with ruff
        if (python.ruff) {
            results.push(await lintWithRuff(filePath));
        }

        // Type check with mypy (optional)
        if (python.mypy) {
            results.push(await typeCheckWithMypy(filePath));
        }
    }

    // Summary
    const successCount = results.filter(r => r.success).length;
    logger.info('Post-edit hook complete', {
        filePath,
        total: results.length,
        successful: successCount,
    });

    return {
        success: true,
        filePath,
        toolName,
        results,
    };
};
