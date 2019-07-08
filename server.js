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
let analysisDb = new sqlite3.Database(config.analysisDb)

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

apiRoutes.get('/getEntrySurveys', (req,res) => {
    let sqlGet = `select grade, branch, status, role, daysAtExercise, deployedPreviously, supportedPreviously, planningAttendance, homeSupport, afsouthSupport, adequateTime, deployInfo, readInstructions from entry_surveys`
    analysisDb.all(sqlGet, [], function(err, rows) {
        if (err) {
            throw err; 
            res.status(400).send({
                 success: false,
                 data: 'Error'
            })
        } else {
            res.status(200).send({
                success: true,
                data: rows
            })
        }
    })
})

apiRoutes.post('/getEntryComment', (req,res) => {
    var commentName = req.body.comment
    let sqlGet = `select ${commentName} from entry_surveys`
    analysisDb.all(sqlGet, [], function(err, rows) {
        if (err) {
            throw err; 
            res.status(400).send({
                 success: false,
                 data: 'Error'
            })
        } else {
            res.status(200).send({
                success: true,
                data: rows
            })
        }
    })
})

apiRoutes.get('/getMidSurveys', (req,res) => {
    let sqlGet = `select grade, branch, status, role, daysAtExercise, deployedPreviously, supportedPreviously, planningAttendance, utilization, training, livingConditions, healthNeeds, timelyEquipment, neededEquipment, planningRating, commNetworks, communicate from camp_surveys`
    analysisDb.all(sqlGet, [], function(err, rows) {
        if (err) {
            throw err; 
            res.status(400).send({
                 success: false,
                 data: 'Error'
            })
        } else {
            res.status(200).send({
                success: true,
                data: rows
            })
        }
    })
})

apiRoutes.get('/getExitSurveys', (req,res) => {
    let sqlGet = `select grade, branch, status, role, daysAtExercise, deployedPreviously, supportedPreviously, planningAttendance, deployAbility, conductingForeign, otherServices, partnerNation, knowledge, utilization, training, deployedEnv, timelyEquipment, neededEquipment, planningRating, commNetworks, communicate, socialExchanges, professionalExchanges, socialRelationships, professionalRelationships, livingConditions, healthNeeds from exit_surveys`
    analysisDb.all(sqlGet, [], function(err, rows) {
        if (err) {
            throw err; 
            res.status(400).send({
                 success: false,
                 data: 'Error'
            })
        } else {
            res.status(200).send({
                success: true,
                data: rows
            })
        }
    })
})

apiRoutes.post('/submitEntrySurvey', (req,res) => {
    //pull values from request
    receivedData = []
    for (var key in req.body) {
        receivedData.push(req.body[key])
    }
    
    //set up sql insert
    let sqlPost = `INSERT INTO entry_surveys 
                    (submitDate, grade, branch, status, role, daysAtExercise, deployedPreviously, supportedPreviously, planningAttendance, religiousPreference, homeSupport, homeSupportComments, afsouthSupport, afsouthSupportComments, adequateTime, adequateTimeComments, deployInfo, deployInfoComments, readInstructions, readInstructionsComments, additionalComments)
                    values (CURRENT_TIMESTAMP, (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?))`
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

apiRoutes.post('/submitCampSurvey', (req,res) => {
    //pull values from request
    receivedData = []
    for (var key in req.body) {
        receivedData.push(req.body[key])
    }
    
    //set up sql insert
    let sqlPost = `INSERT INTO camp_surveys 
                    (submitDate, grade, branch, status, role, daysAtExercise, deployedPreviously, supportedPreviously, planningAttendance, utilization, utilizationComments, training, trainingComments, livingConditions, livingConditionsComments, healthNeeds, healthNeedsComments, timelyEquipment, timelyEquipmentComments, neededEquipment, neededEquipmentComments, planningRating, planningRatingComments, commNetworks, commNetworksComments, communicate, communicateComments, additionalComments)
                    values (CURRENT_TIMESTAMP, (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?))`
    //run insert
    db.run(sqlPost, receivedData, function(err) {
        if (err) {
            console.log(err)
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

apiRoutes.post('/submitExitSurvey', (req,res) => {
    //pull values from request
    receivedData = []
    for (var key in req.body) {
        receivedData.push(req.body[key])
    }
    
    //set up sql insert
    let sqlPost = `INSERT INTO exit_surveys 
                    (submitDate, grade, branch, status, role, daysAtExercise, deployedPreviously, supportedPreviously, planningAttendance, deployAbility, deployAbilityComments, conductingForeign, conductingForeignComments, otherServices, otherServicesComments, partnerNation, partnerNationComments, knowledge, knowledgeComments, utilization, utilizationComments, training, trainingComments, livingConditions, livingConditionsComments, healthNeeds, healthNeedsComments, timelyEquipment, timelyEquipmentComments, neededEquipment, neededEquipmentComments, planningRating, planningRatingComments, commNetworks, commNetworksComments, communicate, communicateComments, socialExchanges, socialExchangesComments, professionalExchanges, professionalExchangesComments, additionalComments)
                    values (CURRENT_TIMESTAMP, (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?), (?))`
    //run insert
    db.run(sqlPost, receivedData, function(err) {
        if (err) {
            //send error back
            console.log(err)
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

