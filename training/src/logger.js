const fs = require('fs');

// Helper function to format and write log messages
function log(level, ...messages) {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] ${level.toUpperCase()} ${messages.join(' ')}`;

    // Open the log file in append mode
    const logStream = fs.createWriteStream('logs.txt', {flags: 'a'});

    // Write the formatted log message to the file
    logStream.write(`${logMessage}\n`);

    // Also output to the console
    console.log(logMessage);
}

// Logger object with info and error methods
const logger = {
    info: (...messages) => log('info', ...messages),
    error: (...messages) => log('error', ...messages),
};

module.exports = logger;
