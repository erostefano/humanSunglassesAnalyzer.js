const fs = require('fs');
const path = require('path');

// Replace with your folder path
const folderPath = 'your-folder-path';

// Function to randomly shuffle an array
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// Function to generate a unique file name to avoid overwriting
function generateUniqueName(folderPath, baseName, extension) {
    let name = `${baseName}${extension}`;
    let counter = 1;

    while (fs.existsSync(path.join(folderPath, name))) {
        name = `${baseName}-${counter}${extension}`;
        counter++;
    }

    return name;
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

    // Rename files
    files.forEach((file, index) => {
        const baseName = `with-sunglasses-${index + 1}`;
        const extension = path.extname(file);
        const oldPath = path.join(folderPath, file);

        // Generate a unique new file name
        const newFileName = generateUniqueName(folderPath, baseName, extension);
        const newPath = path.join(folderPath, newFileName);

        fs.rename(oldPath, newPath, (err) => {
            if (err) {
                console.error(`Error renaming file ${file}:`, err);
            } else {
                console.log(`${file} renamed to ${newFileName}`);
            }
        });
    });
});
