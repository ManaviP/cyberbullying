// Button callers
function goTo(toPage) {
  // Adds the rest of path to the toPage and then go
  window.alert(
    `Website has no real functionalities:\n
    \nAction would take you to: ${toPage}`
  );
}

function openPopup(popupName) {
  if (popupName === "treet") {
    window.alert(
      `Website has no real functionalities:\n
      \nAction would create a new Treet (tweet)`
    );
  }
}

function openMenu(menuName) {
  if (menuName === "accounts-menu") {
    window.alert(
      `Website has no real functionalities:\n
      \nAction would open the menu action (logout and stuff)`
    );
  }
}

// Cool thing using toggle for the Like btn
function changeState(element) {
  element.classList.toggle("liked");
}

// Disable or enable Treex button based on this input
const treex = document.getElementById("treex");

function treexInputChanged(element) {
  if (element.value !== "") {
    // Activate Treex
    treex.classList.remove("disabled");
  } else {
    // Disable Treex
    treex.classList.add("disabled");
  }
  growTreexInput(element); // Resize rows
}

function growTreexInput(element) {
  element.style.height = "5px";
  element.style.height = element.scrollHeight + "px";
}
