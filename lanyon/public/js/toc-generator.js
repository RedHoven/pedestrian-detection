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
    const tag = heading.tagName.toLowerCase();
    hCount++;
    if (hCount <= 2) return;
    
    const level = parseInt(heading.tagName.slice(1)); // e.g. 1, 2, 3
    const text = heading.textContent.trim();
    // Assign an ID if not already present
    const id = heading.id || `${idPrefix}${index}`;
    heading.id = id;
    
    tocItems.push({ id, text, level });
  });
  
  // Build a nested list (<ul>) HTML from the tocItems array.
  function buildTOC(items) {
    let currentLevel = items.length > 0 ? items[0].level : 1;
    let html = "<ul>";
  
    items.forEach(({ id, text, level }) => {
      if (level > currentLevel) {
        // For deeper levels, create nested lists
        html += "<ul>".repeat(level - currentLevel);
      } else if (level < currentLevel) {
        // Close nested lists if we are going back up the level
        html += "</ul>".repeat(currentLevel - level);
      }
      currentLevel = level;
      html += `<li><a href="#${id}">${text}</a></li>`;
    });
  
    html += "</ul>".repeat(currentLevel - 1);
    html += "</ul>";
    return html;
  }
  
  toc.innerHTML = buildTOC(tocItems);
  
  // Scroll spy: highlight the active section and expand its parent branch.
  const sections = tocItems.map(({ id }) => document.getElementById(id));
  
  const activateTOC = () => {
    let activeIndex = -1;
    sections.forEach((section, index) => {
      const rect = section.getBoundingClientRect();
      // Adjust threshold as needed (here 30% of viewport height)
      if (rect.top < window.innerHeight * 0.3) {
        activeIndex = index;
      }
    });
  
    // Remove active classes from all list items
    toc.querySelectorAll("li").forEach(li => li.classList.remove("active"));
  
    if (activeIndex !== -1) {
      const activeId = tocItems[activeIndex].id;
      const activeLink = toc.querySelector(`a[href="#${activeId}"]`);
      if (activeLink) {
        let li = activeLink.closest("li");
        // Mark the active li and all parent li elements as active to expand nested lists.
        while (li) {
          li.classList.add("active");
          li = li.parentElement.closest("li");
        }
      }
    }
  };
  
  window.addEventListener("scroll", activateTOC);
  activateTOC();
});
