// =======================
// get the packages we need ============
// =======================
var express     = require('express')
var http        = require('http')
var https       = require('https')
var fs          = require('fs')
var path        = require('path')
var app         = express()
var bodyParser  = require('body-parser')
var morgan      = require('morgan')
const sqlite3   = require('sqlite3').verbose()
var config      = require('./config') // get our config file
var cors        = require('cors')
var moment      = require('moment')
var formidableMiddleware = require('express-formidable')


// =======================
// configuration =========
// =======================
var port = process.env.PORT || 5005 // used to create, sign, and verify tokens
let db = new sqlite3.Database(config.database)

//CORS
//Requests are only allowed from whitelisted url
// var whitelist = ['http://localhost:8080','https://localhost:8080']
var corsOptions = {
    origin: function (origin, callback){
        // whitelist-test pass
        if (true){//(whitelist.indexOf(origin) !== -1){
            callback(null, true)
        }
        // whitelist-test fail
        else{
            callback(new Error('Not on whitelist'))    
        }
    }
}
app.use(cors(corsOptions))

// use body parser so we can get info from POST and/or URL parameters
app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: true}))
// parse multipart/form-data posts

// use morgan to log requests to the console
app.use(morgan('dev'))

// =======================
// routes ================
// =======================
// basic route

app.get('/', (req, res)=> {
    res.send('Hello! The API is at http://localhost:' + port + '/api')
})

// API ROUTES -------------------
var apiRoutes = express.Router()

apiRoutes.get('/', (req, res)=>{
    res.json({message: 'Welcome to the API ROOT'})
})
apiRoutes.get('/login', (req, res)=>{
    setTimeout(() => {
        res.json({message: 'login'})
    },1000)
})

apiRoutes.get('/spoofGet',(req,res) => {
    let sql1 = `
                    select current_timestamp as date;
                `
    var data = []
    db.serialize(function() {
        db.all(sql1, [], (err, rows) => {
            console.log(rows[0])
            data.push(rows[0])
        })
        db.all(sql1, [], (err, rows) => {
            console.log(rows[0])
            data.push(rows[0])
            console.log(data)
            res.json({
                success: true,
                data: data 
            })
        })
    })
})

apiRoutes.post('/spoofPost',(req,res) => {
    console.log(req)
    if (req) {
        res.status(200).send({
            success: true,
            message: 'Received'
        })
    }
})

apiRoutes.post('/submitCampSurvey', (req,res) => {
    console.log(req.body)
    //pull values from request
    receivedData = []
    for (var key in req.body) {
        receivedData.push(req.body[key])
    }
    console.log(receivedData)
    
    //set up sql insert
    let sqlPost = `INSERT INTO camp_surveys 
                    (submitDate, grade, branch, status, role, skillSets, skillSetsComments, training, trainingComments, deployedEnv, deployedEnvComments, deployInfo, deployInfoComments)
                    values (CURRENT_TIMESTAMP, (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?))`
    //run insert
    db.run(sqlPost, receivedData, function(err) {
        if (err) {
            //send error back
            res.status(400).send({
                success: false,
                message: 'Error attempting to submit data.'
            })
        } else {
            res.status(200).send({
                success: true,
                message: 'Data successfully submitted!'
            })
        }
    })
})

apiRoutes.post('/submitTargetRates', formidableMiddleware(), (req,res) => {
    console.log('beginbegin')
    //pull values from request
    var afsc = req.fields.afsc
    var targetRates = JSON.parse(req.fields.targetRates)
    var person = req.fields.person
    //set up sql insert
    let sqlPost = `INSERT INTO targetRates 
                    (afsc,tier,criteria,percent,submitDate,submittedBy)
                    values `
    //make string for each insert
    let queryValues = "((?), (?), (?), (?), CURRENT_TIMESTAMP, (?))"
    var data = []
    var rowValues = []
    //make one dimensional array for query and data elements
    for (let i = 0; i < targetRates.length; i++) {
        rowValues.push(queryValues)
        data.push(afsc)
        data.push(targetRates[i].tier)
        data.push(targetRates[i].criteria)
        data.push(targetRates[i].percent)
        data.push(person)
    }
    //make large insert string
    sqlPost += rowValues.join(", ");

    db.run(sqlPost, data, function(err) {
        if (err) {
            throw err
        } else {
            res.status(200).send({
                success: true,
                message: 'Data successfully submitted!'
            })
        }
    })
})


app.use('/api', apiRoutes)

// =======================
// start the server ======
// =======================


app.listen(port)
// const options = {
//   key: fs.readFileSync('../tm/localhost.key'),
//   cert: fs.readFileSync('../tm/localhost.crt'),
//   passphrase: '1234'
// };

// https.createServer(options, app).listen(port);
// console.log('Server up at https://localhost:' + port)

console.log('Server up at http://localhost:' + port)

function dtSAStoJS(dtSAS,dtType='DATE'){
  // accepts SAS unformatted DATE or DATETIME
  // dtType should be used to determine the above
  // -315619200000 is equivalent to +new Date(1960,0,1)
  // 86400000 is equivalent to 24h * 60m * 60s * 1000ms
  if(dtType==='DATE'){
    return new Date(-315619200000 + dtSAS * 86400000);
  } else if (dtType==='DATETIME'){
    return new Date(-315619200000 + dtSAS * 1000);
  } else {
    console.log('Unknown dtType value - ' + dtType);
    return null;
  }
};


function formatSASDate(sasdate) {
    if (sasdate){
        var date = new Date(-315619200000 + sasdate * 86400000)

        var year = date.getFullYear();

        var month = (1 + date.getMonth()).toString();
        month = month.length > 1 ? month : '0' + month;

        var day = date.getDate().toString();
        day = day.length > 1 ? day : '0' + day;

        return year + '/' + month + '/' + day;
    }
    else {
        return ""
    }
}
