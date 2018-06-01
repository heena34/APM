// Get dependencies
const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const http = require('http');


//Instatiate express web server
const app = express();


// Parsers for POST data
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));


// Point static path to www
app.use(express.static(path.join(__dirname, 'dist')));


// add API routes
app.use('/app', require('./app'));



// Catch all other routes and return the index file
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist/index.html'));
});


http.createServer(app)
  .listen(5000, () =>
    console.log(`Secure application running on https://localhost:5000`));
