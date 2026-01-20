/**
 * Post-Write Hook for ContextFlow
 *
 * Runs after new file creation:
 * - Ensure __init__.py exists in new directories
 * - Add license header if missing
 *
 * Boris Step 9: "PostToolUse Hook for formatting"
 */

const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
    ensureInit: true,
    addLicenseHeader: false, // Enable if license headers are required
    verbose: process.env.HOOK_VERBOSE === 'true',
};

// License header template (MIT)
const LICENSE_HEADER = `# Copyright (c) 2026 ContextFlow AI
# SPDX-License-Identifier: MIT
`;

/**
 * Logger utility for debugging
 */
const logger = {
    info: (msg, data = {}) => {
        if (CONFIG.verbose) {
            console.log(`[post-write] INFO: ${msg}`, JSON.stringify(data));
        }
    },
    error: (msg, data = {}) => {
        console.error(`[post-write] ERROR: ${msg}`, JSON.stringify(data));
    },
    success: (msg, data = {}) => {
        if (CONFIG.verbose) {
            console.log(`[post-write] SUCCESS: ${msg}`, JSON.stringify(data));
        }
    },
};

/**
 * Ensure __init__.py exists in the directory
 */
function ensureInitFile(filePath) {
    const dir = path.dirname(filePath);
    const initPath = path.join(dir, '__init__.py');

    // Skip if in root or non-src directories
    if (!dir.includes('src') && !dir.includes('contextflow')) {
        logger.info('Skipping __init__.py check (not in src)', { dir });
        return { success: true, skipped: true };
    }

    try {
        if (!fs.existsSync(initPath)) {
            fs.writeFileSync(initPath, '# Auto-generated __init__.py\n');
            logger.success('Created __init__.py', { initPath });
            return { success: true, created: true, path: initPath };
        }
        logger.info('__init__.py already exists', { initPath });
        return { success: true, exists: true, path: initPath };
    } catch (error) {
        logger.error('Failed to create __init__.py', { initPath, error: error.message });
        return { success: false, error: error.message };
    }
}

/**
 * Add license header to file if missing
 */
function addLicenseHeader(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');

        // Check if header already exists
        if (content.includes('SPDX-License-Identifier') || content.includes('Copyright')) {
            logger.info('License header already present', { filePath });
            return { success: true, exists: true };
        }

        // Add header at the beginning
        const newContent = LICENSE_HEADER + '\n' + content;
        fs.writeFileSync(filePath, newContent);
        logger.success('Added license header', { filePath });
        return { success: true, added: true };
    } catch (error) {
        logger.error('Failed to add license header', { filePath, error: error.message });
        return { success: false, error: error.message };
    }
}

/**
 * Main post-write hook function
 */
module.exports = async function postWrite(context) {
    const { filePath } = context;
    const results = [];

    logger.info('Post-write hook triggered', { filePath });

    // Handle Python files
    if (filePath && filePath.endsWith('.py')) {
        // Ensure __init__.py exists
        if (CONFIG.ensureInit) {
            results.push(ensureInitFile(filePath));
        }

        // Add license header if configured
        if (CONFIG.addLicenseHeader) {
            results.push(addLicenseHeader(filePath));
        }
    }

    // Summary
    const successCount = results.filter(r => r.success).length;
    logger.info('Post-write hook complete', {
        filePath,
        total: results.length,
        successful: successCount,
    });

    return {
        success: true,
        filePath,
        results,
    };
};
