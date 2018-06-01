const express = require('express')
const router = express.Router();
const { exec } = require('child_process');
//var multer = require("multer");


// var storage = multer.diskStorage({
//   destination:function(req,file,callback) {
//     callback(null,'./shared/uploads');
//   },
//   filename:function(req,file,callback) {
//     callback(null,file.name);
//   }
// })

router.get('/train', (req, res) => {
    exec('python 23.py', (error, stdout, stderr) => {
        if (error) {
          console.error(`exec error: ${error}`);
          return;
        }
        console.log(`stdout: ${stdout}`);
      });

    res.status(200).send("train");
});

router.get('/test', (req, res) => {
    exec('python 23.py', (error, stdout, stderr) => {
        if (error) {
          console.error(`exec error: ${error}`);
          return;
        }
        console.log(`stdout: ${stdout}`);
      });

    res.status(200).send("test");
});


router.post('/upload', (req, res) => {
  console.log("upload");
  console.log(req.file);

  // upload(req,res,function(err){
  //   if(err) {
  //     return res.end("Something went wrong!.");
  //   }
  //   return res.end("File uploaded successfully!.")
    res.status(200).send("test");
  //});

 
});

module.exports = router;