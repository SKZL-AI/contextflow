/**
 * Pre-Commit Hook for ContextFlow
 *
 * Runs before git commits:
 * - Run tests for changed files
 * - Check for TODO/FIXME markers
 * - Validate imports
 *
 * Boris Step 9: "PostToolUse Hook for formatting"
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs');
const path = require('path');

const execAsync = promisify(exec);

// Configuration
const CONFIG = {
    runTests: true,
    checkTodos: true,
    validateImports: true,
    todoMarkers: ['TODO', 'FIXME', 'HACK', 'XXX'],
    timeout: 60000, // 60 seconds for tests
    verbose: process.env.HOOK_VERBOSE === 'true',
};

/**
 * Logger utility for debugging
 */
const logger = {
    info: (msg, data = {}) => {
        if (CONFIG.verbose) {
            console.log(`[pre-commit] INFO: ${msg}`, JSON.stringify(data));
        }
    },
    warn: (msg, data = {}) => {
        console.warn(`[pre-commit] WARN: ${msg}`, JSON.stringify(data));
    },
    error: (msg, data = {}) => {
        console.error(`[pre-commit] ERROR: ${msg}`, JSON.stringify(data));
    },
    success: (msg, data = {}) => {
        if (CONFIG.verbose) {
            console.log(`[pre-commit] SUCCESS: ${msg}`, JSON.stringify(data));
        }
    },
};

/**
 * Run pytest on changed test files
 */
async function runPytest(testFiles) {
    if (testFiles.length === 0) {
        logger.info('No test files to run');
        return { success: true, skipped: true };
    }

    logger.info('Running pytest', { files: testFiles });

    try {
        const filesArg = testFiles.map(f => `"${f}"`).join(' ');
        const { stdout, stderr } = await execAsync(
            `python -m pytest ${filesArg} -v --tb=short`,
            { timeout: CONFIG.timeout }
        );
        logger.success('Tests passed', { stdout: stdout.slice(0, 500) });
        return { success: true, stdout, stderr };
    } catch (error) {
        logger.error('Tests failed', { error: error.message });
        return { success: false, error: error.message, stdout: error.stdout };
    }
}

/**
 * Check for TODO/FIXME markers in staged files
 */
function checkTodoMarkers(files) {
    const findings = [];

    for (const filePath of files) {
        if (!filePath.endsWith('.py')) continue;
        if (!fs.existsSync(filePath)) continue;

        try {
            const content = fs.readFileSync(filePath, 'utf8');
            const lines = content.split('\n');

            lines.forEach((line, index) => {
                for (const marker of CONFIG.todoMarkers) {
                    if (line.includes(marker)) {
                        findings.push({
                            file: filePath,
                            line: index + 1,
                            marker,
                            text: line.trim().slice(0, 100),
                        });
                    }
                }
            });
        } catch (error) {
            logger.warn('Could not read file for TODO check', { filePath, error: error.message });
        }
    }

    if (findings.length > 0) {
        logger.warn('Found TODO/FIXME markers', { count: findings.length, findings });
    } else {
        logger.info('No TODO/FIXME markers found');
    }

    return { success: true, findings };
}

/**
 * Validate Python imports
 */
async function validateImports(files) {
    const pythonFiles = files.filter(f => f.endsWith('.py'));
    if (pythonFiles.length === 0) {
        return { success: true, skipped: true };
    }

    logger.info('Validating imports', { fileCount: pythonFiles.length });

    const errors = [];

    for (const filePath of pythonFiles) {
        if (!fs.existsSync(filePath)) continue;

        try {
            // Use Python to check syntax and imports
            const { stderr } = await execAsync(
                `python -m py_compile "${filePath}"`,
                { timeout: 10000 }
            );
            if (stderr) {
                errors.push({ file: filePath, error: stderr });
            }
        } catch (error) {
            errors.push({ file: filePath, error: error.message });
        }
    }

    if (errors.length > 0) {
        logger.error('Import validation failed', { errors });
        return { success: false, errors };
    }

    logger.success('Import validation passed');
    return { success: true };
}

/**
 * Main pre-commit hook function
 */
module.exports = async function preCommit(context) {
    const { stagedFiles = [] } = context;
    const results = [];

    logger.info('Pre-commit hook triggered', { fileCount: stagedFiles.length });

    // Filter Python files
    const pythonFiles = stagedFiles.filter(f => f.endsWith('.py'));
    const testFiles = stagedFiles.filter(f =>
        f.includes('/tests/') || f.includes('\\tests\\') || f.startsWith('test_')
    );

    // Run tests for changed test files
    if (CONFIG.runTests && testFiles.length > 0) {
        results.push(await runPytest(testFiles));
    }

    // Check for TODO/FIXME markers
    if (CONFIG.checkTodos) {
        results.push(checkTodoMarkers(pythonFiles));
    }

    // Validate imports
    if (CONFIG.validateImports) {
        results.push(await validateImports(pythonFiles));
    }

    // Check for critical failures
    const failures = results.filter(r => !r.success && !r.skipped);
    const allPassed = failures.length === 0;

    if (!allPassed) {
        logger.error('Pre-commit checks failed', {
            total: results.length,
            failures: failures.length,
        });
    } else {
        logger.success('Pre-commit checks passed', {
            total: results.length,
        });
    }

    return {
        success: allPassed,
        stagedFiles,
        results,
        canProceed: allPassed,
    };
};
