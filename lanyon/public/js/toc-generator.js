document.addEventListener("DOMContentLoaded", () => {
  const toc = document.getElementById("toc");
  if (!toc) return;

  // Get all h1, h2, and h3 headings.
  const headings = Array.from(document.body.querySelectorAll("h1, h2, h3"));

  // Skip the first two h1 elements
  let hCount = 0;
  const tocItems = [];
  const idPrefix = "toc_";

  headings.forEach((heading, index) => {
    hCount++;
    if (hCount <= 2) return;

    const level = parseInt(heading.tagName.slice(1));
    const text = heading.textContent.trim();
    const id = heading.id || `${idPrefix}${index}`;
    heading.id = id;
    tocItems.push({ id, text, level });
  });

  // Build nested TOC
  function buildTOC(items) {
    let html = "";
    let levelStack = [];

    items.forEach(({ id, text, level }) => {
      while (levelStack.length && levelStack[levelStack.length - 1] >= level) {
        html += "</li></ul>";
        levelStack.pop();
      }
      html += `<li><a href="#${id}">${text}</a><ul>`;
      levelStack.push(level);
    });

    while (levelStack.length) {
      html += "</li></ul>";
      levelStack.pop();
    }

    return `<ul>${html}</ul>`;
  }

  toc.innerHTML = buildTOC(tocItems);

  const sections = tocItems.map(({ id }) => document.getElementById(id));

  // Activate ToC items on scroll
  const activateTOC = () => {
    const scrollPosition = window.scrollY || window.pageYOffset;
    const offset = 150;
    let activeItem = null;

    for (let i = 0; i < sections.length; i++) {
      if (sections[i].offsetTop <= scrollPosition + offset) {
        activeItem = tocItems[i];
      } else {
        break;
      }
    }

    // Reset all classes
    toc.querySelectorAll("li").forEach(li => {
      li.classList.remove("active", "expanded");
    });

    if (activeItem) {
      const activeLink = toc.querySelector(`a[href="#${activeItem.id}"]`);
      if (activeLink) {
        // Only mark the current section's li as active
        activeLink.parentElement.classList.add("active");

        // Expand all ancestor li elements
        let parentLi = activeLink.closest("li").parentElement.closest("li");
        while (parentLi) {
          parentLi.classList.add("expanded");
          parentLi = parentLi.parentElement.closest("li");
        }
      }
    }
  };

  window.addEventListener("scroll", activateTOC);
  activateTOC(); // On load
  const toggleBtn = document.getElementById("toc-toggle");
  const tocContainer = document.getElementById("toc-container");

  toggleBtn.addEventListener("click", () => {
    tocContainer.classList.toggle("visible");
    tocContainer.classList.toggle("hidden");
  });

  document.addEventListener("click", (e) => {
    if (
      window.innerWidth <= 768 &&
      tocContainer.classList.contains("visible") &&
      !tocContainer.contains(e.target) &&
      !toggleBtn.contains(e.target)
    ) {
      tocContainer.classList.remove("visible");
      tocContainer.classList.add("hidden");
    }
  });
});
