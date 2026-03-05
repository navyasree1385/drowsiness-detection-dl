const video = document.getElementById("video");

function startCamera(){
    navigator.mediaDevices.getUserMedia({video:true})
    .then(stream=>{
        video.srcObject = stream;
        sendFrames();
    });
}

function sendFrames(){

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    setInterval(()=>{
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        ctx.drawImage(video,0,0);

        let data = canvas.toDataURL("image/jpeg");

        fetch("/predict",{
            method:"POST",
            body:data
        });

    },200);
}