const form = document.getElementById("uploadForm");
const input = document.getElementById("videoInput");
const fileName = document.getElementById("fileName");
const dropzone = document.getElementById("dropzone");
const btn = document.getElementById("analyzeBtn");
const progress = document.getElementById("progress");
const errorBox = document.getElementById("error");
const results = document.getElementById("results");
input.addEventListener("change",()=>{fileName.textContent=input.files[0]?.name || "or drag and drop a clip here";});
["dragenter","dragover"].forEach(ev=>dropzone.addEventListener(ev,e=>{e.preventDefault();dropzone.style.borderColor="#b7f36b";}));
["dragleave","drop"].forEach(ev=>dropzone.addEventListener(ev,e=>{e.preventDefault();dropzone.style.borderColor="";}));
dropzone.addEventListener("drop",e=>{if(e.dataTransfer.files.length){input.files=e.dataTransfer.files;fileName.textContent=input.files[0].name;}});
function addList(id, values, formatter){const el=document.getElementById(id);el.innerHTML="";(Array.isArray(values)?values:[]).forEach(v=>{const li=document.createElement("li");li.textContent=formatter?formatter(v):String(v);el.appendChild(li);});if(!el.children.length){const li=document.createElement("li");li.textContent="No specific observations returned.";el.appendChild(li);}}
function metric(label,value,note){return `<div class="metric"><small>${label}</small><strong>${value}</strong><em>${note||""}</em></div>`;}
form.addEventListener("submit",async e=>{
 e.preventDefault(); errorBox.classList.add("hidden"); results.classList.add("hidden");
 if(!input.files.length){errorBox.textContent="Choose a video first.";errorBox.classList.remove("hidden");return;}
 const data=new FormData(form); btn.disabled=true;progress.classList.remove("hidden");
 try{
  const res=await fetch("/api/analyze",{method:"POST",body:data});
  const payload=await res.json();if(!res.ok)throw new Error(payload.detail||"Analysis failed.");
  const m=payload.metrics||{}, ai=payload.ai||{};
  document.getElementById("reportTitle").textContent=(payload.athlete_name||"Athlete")+" · "+(payload.sport||"Session");
  document.getElementById("aiSource").textContent=ai.source==="gemini_api"?"GEMINI + YOLO":"YOLO + LOCAL FALLBACK";
  document.getElementById("metrics").innerHTML=[
   metric("POSE DETECTION RATE",m.pose_detection_rate_pct==null?"—":m.pose_detection_rate_pct+"%","Frames with a detected person"),
   metric("AVG. KNEE ANGLE",m.average_knee_angle_deg==null?"—":m.average_knee_angle_deg+"°","2D image-plane estimate"),
   metric("AVG. HIP ANGLE",m.average_hip_angle_deg==null?"—":m.average_hip_angle_deg+"°","2D image-plane estimate"),
   metric("TRACKING CONFIDENCE",m.average_keypoint_confidence==null?"—":(m.average_keypoint_confidence*100).toFixed(0)+"%","Mean model keypoint confidence")
  ].join("");
  const vid=document.getElementById("resultVideo");vid.src=payload.video_url+"?v="+Date.now();vid.load();
  document.getElementById("assessment").textContent=ai.overall_assessment||"No assessment returned.";
  const score=ai.performance_score;document.getElementById("scoreWrap").classList.toggle("hidden",score==null);if(score!=null)document.getElementById("score").textContent=score;
  addList("strengths",ai.strengths);addList("improvements",ai.areas_to_improve);addList("technical",ai.technical_observations);addList("drills",ai.recommended_drills);addList("tactics",ai.tactical_insights);
  addList("moments",ai.key_moments,x=>`${x.timestamp||"Time n/a"} — ${x.observation||""} (${x.confidence||"unspecified"} confidence)`);
  const limits=[...(ai.limitations||[]),m.measurement_note||"",ai.gemini_error?("Gemini API issue: "+ai.gemini_error):""].filter(Boolean);
  document.getElementById("limitations").textContent=limits.join(" · ");
  results.classList.remove("hidden");results.scrollIntoView({behavior:"smooth"});
 }catch(err){errorBox.textContent=err.message;errorBox.classList.remove("hidden");}
 finally{btn.disabled=false;progress.classList.add("hidden");}
});
