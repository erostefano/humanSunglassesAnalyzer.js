const fs = require('fs');
const path = require('path');

// Replace with your folder path
const folderPath = 'without-sunglasses';

// Function to randomly shuffle an array
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

fs.readdir(folderPath, (err, files) => {
    if (err) {
        console.error('Error reading directory:', err);
        return;
    }

    // Shuffle the files array
    shuffle(files);

    // Filter only files (exclude folders)
    files = files.filter(file => fs.lstatSync(path.join(folderPath, file)).isFile());

    // First pass: Rename to temporary unique names
    files.forEach((file, index) => {
        const tempFileName = `tempfile-${index}${path.extname(file)}`;
        const oldPath = path.join(folderPath, file);
        const tempPath = path.join(folderPath, tempFileName);

        fs.renameSync(oldPath, tempPath);
    });

    // Second pass: Rename from temporary names to final names
    files.forEach((file, index) => {
        const tempFileName = `tempfile-${index}${path.extname(file)}`;
        const finalFileName = `without-sunglasses-${index + 1}${path.extname(file)}`;
        const tempPath = path.join(folderPath, tempFileName);
        const finalPath = path.join(folderPath, finalFileName);

        fs.renameSync(tempPath, finalPath);
    });

    console.log("Files have been renamed successfully without any loss.");
});
