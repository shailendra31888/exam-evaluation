const loginLink = document.querySelector('.login-link');
const wrapper = document.querySelector('.wrapper');
const RegisterLink = document.querySelector('.register-link');
var form1 = document.querySelector('.form-login');
var form2 = document.querySelector('.form-register');

RegisterLink.addEventListener('click', ()=>{
    console.log("Got class");
    form1.classList.add("left-shift");
    wrapper.classList.add("height")
    form2.classList.add("left-shift")
});

loginLink.addEventListener('click', ()=>{
    console.log("Got class");
    form1.classList.remove("left-shift");
    form2.classList.remove("left-shift");
    wrapper.classList.remove("height");
})