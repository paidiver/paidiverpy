function toggleMetadata(stepIndex, randomId) {
  var metadata = document.getElementById("ppy-images-metadata-" + randomId + "-" + stepIndex);
  var arrow = document.getElementById("ppy-images-arrow-" + randomId + "-" + stepIndex);
  if (metadata.style.display === "none") {
      metadata.style.display = "block";
      arrow.innerHTML = "▼";
      arrow.classList.remove("ppy-font-color");
      arrow.classList.add("ppy-font-color-brown");
  } else {
      metadata.style.display = "none";
      arrow.innerHTML = "►";
      arrow.classList.remove("ppy-font-color-brown");
      arrow.classList.add("ppy-font-color");
  }
}
function toggleImage(imageId, randomId) {
  var img = document.getElementById(imageId);
  var arrow = document.getElementById("ppy-images-arrow-" + randomId + "-" + imageId);
  if (img.style.display === "none") {
      img.style.display = "block";
      arrow.innerHTML = "▼";
      arrow.classList.remove("ppy-font-color");
      arrow.classList.add("ppy-font-color-brown");
  } else {
      img.style.display = "none";
      arrow.innerHTML = "►";
      arrow.classList.remove("ppy-font-color-brown");
      arrow.classList.add("ppy-font-color");
  }
}
function showMore(stepIndex) {
  var moreImages = document.getElementById("ppy-more-images-" + stepIndex);
  var showMoreButton = document.getElementById("ppy-show-more-button-" + stepIndex);
  var hideButton = document.getElementById("ppy-hide-button-" + stepIndex);
  moreImages.style.display = "flex";
  showMoreButton.style.display = "none";
  hideButton.style.display = "inline-block";
}
function hide(stepIndex) {
  var moreImages = document.getElementById("ppy-more-images-" + stepIndex);
  var showMoreButton = document.getElementById("ppy-show-more-button-" + stepIndex);
  var hideButton = document.getElementById("ppy-hide-button-" + stepIndex);
  moreImages.style.display = "none";
  showMoreButton.style.display = "inline-block";
  hideButton.style.display = "none";
}


function showParameters(id, randomId) {
  var currentTarget = document.getElementById('ppy-pipeline-' + randomId + '-' + id);
  var square = document.getElementsByClassName('ppy-pipeline-' + randomId);
  var allParams = document.getElementsByClassName('ppy-pipeline-parameters-' + randomId);
  var selectedParams = document.getElementById('ppy-pipeline-parameters-' + randomId + '-' + id);
  var idWasVisible = false;
  if (selectedParams) {
      var idWasVisible = selectedParams.style.display === 'block';
  }
  for (var i = 0; i < square.length; i++) {
      square[i].classList.remove('ppy-font-color-brown');
      square[i].classList.add('ppy-font-color');
  }
  for (var i = 0; i < allParams.length; i++) {
      allParams[i].style.display = 'none';
  }
  // Show the selected parameter section
  if (selectedParams) {
      if (idWasVisible) {
          selectedParams.style.display = 'none';
          currentTarget.classList.remove('ppy-font-color-brown');
          currentTarget.classList.add('ppy-font-color');
      } else {
          selectedParams.style.display = 'block';
          currentTarget.classList.remove('ppy-font-color');
          currentTarget.classList.add('ppy-font-color-brown');
      }
  }
}

hljs.highlightAll();
