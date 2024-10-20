const express = require('express')
const mongoose = require('mongoose')
const path = require('path')
const port = 4000

const app = express();
app.use(express.static(__dirname))
app.use(express.urlencoded({extended:true}))
mongoose.connect('mongodb://127.0.0.1:27017/stocks')
const db = mongoose.connection
db.once('open',()=>{
    console.log("mongodb connection sucessful")
})

const userSchema = new mongoose.Schema({
    email:String
})
const users = mongoose.model("data",userSchema)
app.get('/',(req,res)=>{ 
    res.sendFile(path.join(__dirname,'index.html'))
})
app.post('/post',async (req,res) =>{
    const {email} = req.body
    const user = new users({
        email
    })
    await user.save()
    console.log(user)
    res.send("form submission successful")
})


app.listen(port,()=>{
    console.log("Server starter")
})


