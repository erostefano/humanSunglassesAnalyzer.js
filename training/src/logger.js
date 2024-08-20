const fs = require('fs');

function log(...messages) {
    const logStream = fs.createWriteStream('logs.txt', {flags: 'a'});
    console.log(messages.join(';'))
    logStream.write(`${messages.join(';')}\n`);
}

module.exports = {log};

