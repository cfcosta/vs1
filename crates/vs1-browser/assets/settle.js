((action) => {
  const page = __SNAPSHOT__;
  if (!page) return null;
  // Node identities are document-local: never mistake a new document's node for
  // the opener from the previous page after navigation.
  const field =
    action.time_origin === page.marker[0] ? window.__jevFast?.nodes.get(action.node) : null;
  const openingMenu =
    action.kind === "click" &&
    action.expanded !== "true" &&
    field?.isConnected &&
    !field.matches("input,textarea") &&
    ["listbox", "menu", "tree", "grid", "dialog", "true"].includes(
      field.getAttribute("aria-haspopup"),
    );
  // Some sites point aria-controls at an empty placeholder, so use the actual
  // visible snapshot rather than assuming the popup is inside that element.
  const menuVisible = page.actions.some((a) =>
    ["option", "menuitem", "menuitemradio", "gridcell"].includes(a.role),
  );
  const dialogVisible = [...document.querySelectorAll('dialog[open],[role="dialog"]')].some((e) => {
    const r = e.getBoundingClientRect();
    return (
      !e.closest('[aria-hidden="true"],[inert]') &&
      e.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }) &&
      r.width > 0 &&
      r.height > 0 &&
      r.bottom > 0 &&
      r.right > 0 &&
      r.top < innerHeight &&
      r.left < innerWidth
    );
  });
  const usable =
    page.text.trim().length > 0 ||
    page.actions.some((a) => ["click", "fill", "select"].includes(a.kind));
  return {
    page,
    ready: usable && (!openingMenu || menuVisible || dialogVisible),
    waiting_for_menu: Boolean(openingMenu && !menuVisible && !dialogVisible),
  };
})(__ACTION__);
