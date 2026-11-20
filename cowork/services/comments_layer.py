"""On-artifact comment marker layer for the in-app preview iframe.

Ported from mindshub_services' ``comments_ui.py`` (the redesigned commenter
chrome, ENG-472/PR #157) — same interaction engine:

  * element selection: deepest element under the pointer, gap-space geometric
    hit-test, nearest-sibling snap, page container as the deliberate fallback;
    Alt (⌥) steps the selection up to parents
  * custom pin cursor in comment mode
  * pin markers: one node per thread in a single container, measure-then-write
    rAF-batched positioning, Resize/MutationObserver re-sync, staggered
    entrance, cascade for multiple pins on one anchor, active pin vivid /
    others dimmed while a thread is open
  * thread & composer popovers: pill input that grows into a box (hidden
    mirror measurement), replies append in place, edit/delete via the header
    "…" menu, viewport-bound with internal scroll

Transport differs from the published shell: there the wrapper and iframe are
same-origin and the layer calls the comments API directly. In Cowork the
preview iframe is served by cowork-server while the renderer lives on a
different origin (``file://`` in Electron), so:

  * we inject the layer here, at serve time, into the artifact's HTML — the
    only way to get a script into a cross-origin iframe; and
  * the layer is credential-free: it owns DOM concerns only and exchanges
    data with the renderer purely over ``postMessage``. The renderer holds
    the MindsHub creds and performs every comments API call (see
    ``useArtifactComments`` + ``useArtifactCommentLayer`` on the client).

postMessage contract (both sides tag messages ``source:"anton-comments"``):
  layer -> parent : {type:"ready"}
                    {type:"create", selector, text}
                    {type:"reply", id, text}
                    {type:"status", id, status}
                    {type:"edit", id, text}                    // edit root comment
                    {type:"delete", id}                        // delete whole thread
                    {type:"edit-reply", id, replyId, text}
                    {type:"delete-reply", id, replyId}
                    {type:"count", count}
                    {type:"mode", active}
                    {type:"shortcut", key}                     // e.g. "c" (toolbar toggle)
  parent -> layer : {type:"list", comments:[...], viewer, markersVisible}
                                                   // full normalized set + viewer id
                                                   // + pin visibility (atomic with data)
                    {type:"markers", visible}      // toggle pin visibility
                    {type:"enter-mode"|"exit-mode"}
                    {type:"close-pop"}             // discard an in-progress popover
                    {type:"focus", commentId}
                    {type:"hl-on"|"hl-off", commentId}

The normalized comment shape the layer renders (parent produces it):
  {id, selector, status, author, author_user_id, text, edited_at, created_at (epoch s),
   replies:[{id, author, author_user_id, text, edited_at, created_at}]}
Owner-only edit/delete affordances render when author_user_id === viewer.user_id.
"""

import json
import re

# The layer only arms itself when the renderer opts in via this query flag on
# the iframe's entry-document URL, so ordinary previews are untouched.
ACTIVATION_PARAM = "__antonComments"

# The overlays load no web font: Inter is used when the page or the OS already
# has it, system fonts otherwise. On web the preview is a srcdoc that inherits
# the app shell's CSP, which blocks remote stylesheets and fonts, and the
# preview shim would report that block as the artifact's own error.
FONT_STACK = "'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"

# NewShadow/modal-sm — dropdown menu
SHADOW_SM = ("0 0 0 0.5px rgba(39,39,42,0.1),0 1px 1px -0.5px rgba(0,0,0,0.04),"
             "0 3px 3px -1.5px rgba(0,0,0,0.04),0 6px 6px -3px rgba(0,0,0,0.04),"
             "0 12px 12px -6px rgba(0,0,0,0.04)")
# NewShadow/modal-md — thread popover
SHADOW_MD = ("0 0 0 0.5px rgba(39,39,42,0.1),0 1px 1px -0.5px rgba(0,0,0,0.04),"
             "0 3px 3px 0 rgba(0,0,0,0.04),0 6px 6px 0 rgba(0,0,0,0.04),"
             "0 12px 12px 0 rgba(0,0,0,0.04)")

ICONS = {
    # Popover close, 20px, 1.5px stroke
    "x20": ('<svg width="20" height="20" viewBox="0 0 20 20" fill="none">'
            '<path d="M5.63112 14.3689L9.99999 10M14.3689 5.63113L9.99999 10M9.99999 '
            '10L5.63112 5.63113M9.99999 10L14.3689 14.3689" stroke="currentColor" '
            'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    # Resolve check-circle, 16px
    "check": ('<svg width="16" height="16" viewBox="0 0 16 16" fill="none">'
              '<path d="M4.66667 8.33333L6.66667 10.3333L11.3333 5.66667" '
              'stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>'
              '<path d="M8 14.6667C11.6819 14.6667 14.6667 11.6819 14.6667 8C14.6667 '
              '4.3181 11.6819 1.33333 8 1.33333C4.3181 1.33333 1.33333 4.3181 1.33333 '
              '8C1.33333 11.6819 4.3181 14.6667 8 14.6667Z" stroke="currentColor" '
              'stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    # Overflow dots — drawn horizontal
    "dots": ('<svg width="12" height="12" viewBox="0 0 12 12" fill="none">'
             '<path d="M2 6.5C2.27614 6.5 2.5 6.27614 2.5 6C2.5 5.72386 2.27614 5.5 2 '
             '5.5C1.72386 5.5 1.5 5.72386 1.5 6C1.5 6.27614 1.72386 6.5 2 6.5Z" '
             'fill="currentColor" stroke="currentColor" stroke-width="0.75"/>'
             '<path d="M6 6.5C6.27614 6.5 6.5 6.27614 6.5 6C6.5 5.72386 6.27614 5.5 6 '
             '5.5C5.72386 5.5 5.5 5.72386 5.5 6C5.5 6.27614 5.72386 6.5 6 6.5Z" '
             'fill="currentColor" stroke="currentColor" stroke-width="0.75"/>'
             '<path d="M10 6.5C10.2761 6.5 10.5 6.27614 10.5 6C10.5 5.72386 10.2761 5.5 '
             '10 5.5C9.72386 5.5 9.5 5.72386 9.5 6C9.5 6.27614 9.72386 6.5 10 6.5Z" '
             'fill="currentColor" stroke="currentColor" stroke-width="0.75"/></svg>'),
    # Send arrow-up, 16px, 1.5px stroke
    "arrowUp": ('<svg width="16" height="16" viewBox="0 0 16 16" fill="none">'
                '<path d="M8.16667 12.3333V4M4.16667 8L8.16667 4L12.1667 8" '
                'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
                'stroke-linejoin="round"/></svg>'),
    # Edit (pencil), 16px, 1px stroke
    "pencil": ('<svg width="16" height="16" viewBox="0 0 16 16" fill="none">'
               '<path d="M11.333 2.333a1.414 1.414 0 012 2L5.667 12 3 12.667 3.667 '
               '10l7.666-7.667Z" stroke="currentColor" stroke-linecap="round" '
               'stroke-linejoin="round"/></svg>'),
    # Delete (trash), 16px, 1px stroke
    "trash": ('<svg width="16" height="16" viewBox="0 0 16 16" fill="none">'
              '<path d="M2 4H14M5.33333 4V2.66667C5.33333 2.29848 5.63181 2 6 '
              '2H10C10.3682 2 10.6667 2.29848 10.6667 2.66667V4M6.66667 7.33333V10.6667'
              'M9.33333 7.33333V10.6667M3.33333 4L4 12.6667C4 13.4031 4.59695 14 5.33333 '
              '14H10.6667C11.403 14 12 13.4031 12 12.6667L12.6667 4" '
              'stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    # Broken-chain-link (orphaned comment notice), 16px grid, 1.5px stroke
    "unlink": ('<svg width="16" height="16" viewBox="0 0 16 16" fill="none">'
               '<path d="M9.7 4.1l1.05-1.05a2.62 2.62 0 013.7 3.7L13.4 7.8'
               'M6.3 11.9l-1.05 1.05a2.62 2.62 0 01-3.7-3.7L2.6 8.2'
               'M4.3 1.4l.35 1.4M1.4 4.3l1.4.35M11.7 14.6l-.35-1.4M14.6 11.7l-1.4-.35" '
               'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
               'stroke-linejoin="round"/></svg>'),
}

# Comment-mode cursor: the marker pin shape in white with a plus, hotspot at
# the tail (bottom-left). Native 24px viewBox so the hairline border renders
# crisp instead of a downscaled blur.
_CURSOR_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'>"
    "<path d='M0.75 12C0.75 5.79 5.79 0.75 12 0.75C18.21 0.75 23.25 5.79 23.25 12"
    "C23.25 18.21 18.21 23.25 12 23.25H2.9C1.71 23.25 0.75 22.29 0.75 21.1V12Z' "
    "fill='white' stroke='rgba(0,0,0,0.3)' stroke-width='1.25'/>"
    "<path d='M12 7.8v8.4M7.8 12h8.4' stroke='%23202021' stroke-width='1.75' "
    "stroke-linecap='round'/></svg>"
)
CURSOR_CSS = ('url("data:image/svg+xml,'
              + _CURSOR_SVG.replace("<", "%3C").replace(">", "%3E").replace("#", "%23")
              + '") 1 23, crosshair')

_ICONS_JS = json.dumps(ICONS)

_LAYER_CSS = (
    # The layer is injected INTO the artifact document, so the page's own
    # global rules bleed into our controls. Neutralize the risky props;
    # !important beats host rules regardless of specificity.
    ".act-pop,.act-pop *,.act-marker,.act-marker *,.act-menu,.act-menu *"
    "{box-sizing:border-box!important;}"
    ".act-pop button,.act-pop textarea,.act-menu button{margin:0!important;padding:0;"
    "border:none;background:none;font-family:inherit;"
    "min-width:0!important;min-height:0!important;}"
    ".act-pop svg,.act-marker svg,.act-menu svg{flex:0 0 auto!important;}"
    # The menu/pop live in the artifact document — pin the exact glyph and text
    # metrics with !important so host rules can't inflate them.
    ".act-menu svg{width:16px!important;height:16px!important;}"
    ".act-menu-item{height:32px!important;line-height:normal!important;}"
    ".act-menu-item span{font-size:14px!important;line-height:24px!important;"
    "font-weight:400!important;margin:0!important;}"

    # Hover highlight while placing a comment — snaps instantly between
    # elements (only its visibility fades). border-box so the 2px border
    # draws INSIDE the element's rect.
    ".act-hl{position:absolute;pointer-events:none;border:2px solid #146573;"
    "background:rgba(20,101,115,0.06);z-index:2147483639;border-radius:3px;"
    "box-sizing:border-box!important;transition:opacity .08s ease-out;}"

    # Comment-mode cursor on the page — but NOT inside our own UI, where the
    # normal pointer/text cursors apply so X/send/menu feel clickable.
    "body.act-mode,body.act-mode *{cursor:" + CURSOR_CSS + "!important;}"
    "body.act-mode .act-pop,body.act-mode .act-pop *,body.act-mode .act-menu,"
    "body.act-mode .act-menu *{cursor:default!important;}"
    "body.act-mode .act-pop button,body.act-mode .act-menu-item,"
    "body.act-mode .act-send,body.act-mode .act-marker,"
    "body.act-mode .act-marker *{cursor:pointer!important;}"
    "body.act-mode .act-pop textarea{cursor:text!important;}"

    # Marker layer: one container, faded in only after first placement so
    # markers never flash at (0,0) while the doc is still settling.
    "#act-mk{position:absolute;top:0;left:0;width:0;height:0;z-index:2147483640;"
    "opacity:0;transition:opacity .25s ease;}"
    "#act-mk.act-ready{opacity:1;}"
    "#act-mk.act-off{display:none;}"

    # Pin marker: teal droplet, sharp bottom-left tail. The outer node owns
    # position (translate, batched writes); the inner pin owns the
    # hover/entrance scale so the two transforms never fight.
    ".act-marker{position:absolute;top:0;left:0;width:28px;height:28px;"
    "pointer-events:auto;cursor:pointer;will-change:transform;contain:layout style;}"
    ".act-marker:hover{z-index:1;}"
    ".act-marker.act-active{z-index:2;}"
    ".act-pin{width:28px;height:28px;border-radius:14px 14px 14px 4px;"
    "background:#146573;box-shadow:inset 0 0 0 1px rgba(0,0,0,0.26),"
    "0 2px 8px rgba(0,0,0,0.18);color:#fff;display:flex;align-items:center;"
    "justify-content:center;font-family:" + FONT_STACK + ";font-weight:500;"
    "font-size:13px;line-height:normal;transform-origin:0 100%;"
    "transition:transform .16s cubic-bezier(.2,.8,.2,1);}"
    ".act-pin.act-2d{font-size:10px;}"
    ".act-marker:hover .act-pin{transform:scale(1.07);}"
    # Never enter from scale(0) — nothing real appears from nothing.
    "@keyframes act-pin-in{0%{transform:scale(.5);opacity:0}"
    "60%{transform:scale(1.08);opacity:1}100%{transform:scale(1)}}"
    ".act-marker.act-new .act-pin{animation:act-pin-in .35s cubic-bezier(.2,.8,.2,1);}"
    # In-pin spinner while a just-submitted comment is in flight to the server.
    ".act-spin{display:block;width:12px;height:12px;border-radius:50%;"
    "border:2px solid rgba(255,255,255,0.35);border-top-color:#fff;"
    "animation:act-rot .7s linear infinite;}"
    "@keyframes act-rot{to{transform:rotate(360deg)}}"
    # While a thread is open, its pin stays vivid (and a touch larger); every
    # other pin greys out so the active conversation is unmistakable.
    "#act-mk.act-focus .act-marker:not(.act-active) .act-pin{background:#B4B4B6;"
    "color:rgba(255,255,255,0.75);"
    "box-shadow:inset 0 0 0 1px rgba(0,0,0,0.12),0 1px 4px rgba(0,0,0,0.1);}"
    "#act-mk.act-focus .act-marker.act-active .act-pin{transform:scale(1.07);}"

    # Thread / composer popover: white card, 329px, modal-md shadow. Flex
    # column so a height-capped card shrinks its scrollable messages section
    # instead of overflowing the viewport.
    ".act-pop{position:absolute;z-index:2147483642;width:329px;background:#fff;"
    "border-radius:12px;box-shadow:" + SHADOW_MD + ";font-family:" + FONT_STACK + ";"
    "display:flex;flex-direction:column;"
    "transform-origin:top left;animation:act-pop-in .28s cubic-bezier(.16,1,.3,1);}"
    "@keyframes act-pop-in{0%{opacity:0;transform:scale(.96) translateY(6px)}"
    "100%{opacity:1;transform:none}}"
    ".act-pop-head{display:flex;align-items:center;justify-content:space-between;"
    "background:#FCFCFC;border-bottom:0.5px solid rgba(39,39,42,0.1);"
    "padding:8px 8px 8px 10px;border-radius:12px 12px 0 0;flex-shrink:0;}"
    ".act-pop-title{font-size:12px;font-weight:500;line-height:20px;color:#202021;}"
    ".act-pop-tools{display:flex;align-items:center;gap:4px;}"
    ".act-pop-tools button{width:22px;height:22px;border-radius:4px;display:flex;"
    "align-items:center;justify-content:center;cursor:pointer;color:#828285;"
    "transition:color .15s ease,background .15s ease,transform .15s ease;}"
    ".act-pop-tools button:hover{color:#202021;background:rgba(39,39,42,0.08);}"
    ".act-pop-tools button:active{transform:scale(.9);}"
    # The trigger keeps its pressed look for as long as its menu is open.
    ".act-pop-tools .act-t-menu.act-on{color:#202021;background:rgba(39,39,42,0.12);}"
    ".act-pop-tools .act-resolved{color:#146573;}"
    ".act-pop-body{padding:8px;display:flex;flex-direction:column;gap:8px;"
    "flex:1;min-height:0;}"
    ".act-reply{flex-shrink:0;}"
    ".act-msgs{display:flex;flex-direction:column;gap:8px;max-height:40vh;"
    "flex:1 1 auto;min-height:0;"
    "overflow-y:auto;overscroll-behavior:contain;}"
    ".act-msgs::-webkit-scrollbar{width:6px;}"
    ".act-msgs::-webkit-scrollbar-track{background:transparent;}"
    ".act-msgs::-webkit-scrollbar-thumb{background:rgba(39,39,42,0.18);"
    "border-radius:3px;}"
    ".act-msgs::-webkit-scrollbar-thumb:hover{background:rgba(39,39,42,0.32);}"
    ".act-msg{display:flex;flex-direction:column;gap:8px;padding:0 0 8px;}"
    ".act-msg-head{display:flex;align-items:center;gap:8px;}"
    ".act-msg-who{display:flex;align-items:center;gap:4px;min-width:0;}"
    ".act-msg-name{font-size:12px;font-weight:500;line-height:20px;color:#202021;"
    "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}"
    ".act-msg-date{font-size:12px;font-weight:400;line-height:16px;color:#69696B;"
    "white-space:nowrap;}"
    ".act-msg-text{padding-left:20px;font-size:14px;font-weight:400;line-height:20px;"
    "color:#202021;white-space:pre-wrap;word-break:break-word;}"
    ".act-edited{font-size:11px;font-weight:400;color:#828285;}"
    # Own replies: edit/delete revealed on hover, right-aligned in the head row.
    ".act-msg-acts{display:none;align-items:center;gap:2px;margin-left:auto;}"
    ".act-msg:hover .act-msg-acts{display:flex;}"
    ".act-msg-acts button{width:20px;height:20px;border-radius:4px;display:flex;"
    "align-items:center;justify-content:center;cursor:pointer;color:#828285;"
    "transition:color .15s ease,background .15s ease;}"
    ".act-msg-acts button:hover{color:#202021;background:rgba(39,39,42,0.08);}"
    ".act-msg-acts svg{width:13px!important;height:13px!important;}"
    ".act-editbox{margin-top:4px;padding-left:20px;display:flex;}"
    ".act-editbox .act-inwrap{flex:1;}"
    ".act-div{border-top:0.5px solid rgba(39,39,42,0.1);}"
    # Reply / compose row: pill input that grows into a rounded-12 box.
    ".act-reply{display:flex;gap:4px;align-items:flex-start;}"
    ".act-reply .act-av{margin-top:2px;}"
    # No radius transition — the pill <-> box switch is instant.
    ".act-inwrap{flex:1;display:flex;align-items:flex-end;gap:2px;background:#fff;"
    "border:0.5px solid rgba(39,39,42,0.15);border-radius:42px;padding:8px 4px 8px 8px;"
    "transition:border-color .15s ease;min-width:0;}"
    # Any input (composer AND reply) grows from pill into a rounded box:
    # text on top, hairline divider, send button on its own bottom-right row.
    # Flex `order` slots the ::after divider between the textarea (0) and the
    # send button (2).
    ".act-inwrap.act-multi{flex-direction:column;align-items:flex-end;gap:8px;"
    "padding:12px;border-radius:16px;}"
    # flex:1 must not carry into the column layout — there it becomes
    # flex-basis:0 for HEIGHT and collapses the textarea to one line no
    # matter what height we set (verified in headless Chrome).
    ".act-inwrap.act-multi textarea{flex:0 0 auto;align-self:stretch;width:100%;}"
    ".act-inwrap.act-multi::after{content:'';order:1;display:block;width:100%;"
    "border-top:0.5px solid rgba(39,39,42,0.1);}"
    ".act-inwrap.act-multi .act-send{order:2;}"
    # Images (ENG-4410): pasted ones wait in a strip above the text, inside the
    # box; a message's images sit under its text. Clicking one enlarges it.
    ".act-pend{order:-1;display:flex;flex-wrap:wrap;gap:6px;align-self:stretch;}"
    ".act-pend:empty{display:none;}"
    ".act-pend-item{position:relative;width:48px;height:48px;}"
    ".act-pend-item img{width:48px;height:48px;object-fit:cover;border-radius:6px;"
    "border:0.5px solid rgba(39,39,42,0.15);display:block;}"
    ".act-pend-x{position:absolute;top:-6px;right:-6px;width:18px;height:18px;border-radius:50%;"
    "border:0;padding:0;background:#202021;color:#fff;font-size:12px;line-height:18px;cursor:pointer;}"
    ".act-imgs{display:flex;flex-wrap:wrap;gap:6px;padding:6px 0 0 20px;}"
    ".act-img{display:block;width:64px;height:64px;border-radius:6px;background:rgba(39,39,42,0.06);"
    "overflow:hidden;cursor:zoom-in;}"
    ".act-img img{width:100%;height:100%;object-fit:cover;display:block;}"
    ".act-img.act-big{width:100%;height:auto;cursor:zoom-out;}"
    ".act-img.act-big img{height:auto;max-height:320px;object-fit:contain;}"
    ".act-inwrap textarea::-webkit-scrollbar{width:6px;}"
    ".act-inwrap textarea::-webkit-scrollbar-track{background:transparent;}"
    ".act-inwrap textarea::-webkit-scrollbar-thumb{background:rgba(39,39,42,0.18);"
    "border-radius:3px;}"
    ".act-inwrap:focus-within{border-color:rgba(39,39,42,0.3);}"
    ".act-inwrap textarea{flex:1;min-width:0;outline:none;resize:none;"
    "font-family:inherit;font-size:14px;font-weight:400;line-height:24px;"
    "color:#111115;padding:0 4px!important;max-height:240px;background:none;}"
    ".act-inwrap textarea::placeholder{color:#828285;}"
    ".act-send{width:24px;height:24px;border-radius:50%;display:flex;flex-shrink:0;"
    "align-items:center;justify-content:center;cursor:pointer;color:#111115;"
    "opacity:.45;transition:background .15s ease,opacity .15s ease,transform .15s ease;}"
    ".act-send:active{transform:scale(.92);}"
    ".act-send.act-on{background:#146573;color:#fff;opacity:1;}"
    # Orphaned-thread notice (element gone / changed / not visible here).
    ".act-orphan{display:flex;gap:8px;align-items:flex-start;padding:10px 10px 2px;"
    "color:#69696B;font-size:12px;font-weight:400;line-height:16px;}"
    ".act-orphan svg{flex-shrink:0;width:16px!important;height:16px!important;"
    "color:#828285;}"
    # New-comment composer (pin + floating pill, no card/header): the pill IS
    # the surface; typing grows it into a rounded box with the send button on
    # its own bottom-right row.
    ".act-pop.act-compose{background:transparent;box-shadow:none;border-radius:0;"
    "width:300px;}"
    ".act-compose .act-inwrap{background:#fff;border:0.5px solid rgba(39,39,42,0.15);"
    "box-shadow:" + SHADOW_MD + ";padding:8px 8px 8px 12px;}"
    # Avatar (16px, initials fallback)
    ".act-av{width:16px;height:16px;border-radius:50%;flex-shrink:0;display:flex;"
    "align-items:center;justify-content:center;color:#fff;font-size:8px;"
    "font-weight:600;box-shadow:0 0 0 0.5px rgba(39,39,42,0.1);"
    "font-family:" + FONT_STACK + ";}"
    # Dropdown (Edit / Delete) inside the popover header.
    ".act-menu{position:absolute;z-index:2147483643;width:145px;background:#fff;"
    "border-radius:8px;box-shadow:" + SHADOW_SM + ";padding:4px 0;"
    "font-family:" + FONT_STACK + ";animation:act-pop-in .18s cubic-bezier(.16,1,.3,1);}"
    ".act-menu-item{display:flex;align-items:center;gap:2px;height:32px;margin:0 4px;"
    "padding:0 6px;border-radius:4px;cursor:pointer;color:#111115;"
    "transition:background .12s ease;}"
    ".act-menu-item:hover{background:rgba(39,39,42,0.06);}"
    ".act-menu-item .act-mi-ic{width:20px;height:20px;display:flex;align-items:center;"
    "justify-content:center;color:#202021;}"
    ".act-menu-item span{font-size:14px;font-weight:400;line-height:24px;padding:0 4px;}"
    "@media (prefers-reduced-motion:reduce){.act-pop,.act-menu,"
    ".act-marker.act-new .act-pin{animation:none!important;}}"
)

LAYER_JS = r"""
(function(){
  if (window.__antonCommentsLayer) return;
  window.__antonCommentsLayer = true;
  var ICONS = __ICONS__;
  var mode = false, comments = [], hoverEl = null, pop = null, menu = null;
  var meViewer = null;  // parent-echoed {user_id,email}; gates edit/delete UI

  var css = document.createElement('style');
  css.textContent = __LAYER_CSS__;
  document.head.appendChild(css);

  var hl = document.createElement('div');
  hl.className = 'act-hl'; hl.style.opacity = '0';
  document.body.appendChild(hl);

  // Single marker container: batched positioning, one fade-in (no flash).
  var mk = document.createElement('div');
  mk.id = 'act-mk';
  document.body.appendChild(mk);

  // Cross-origin: the parent (renderer) lives on a different origin than this
  // iframe, so we can't target a specific origin — post to '*' and let both
  // sides gate on the source tag.
  function send(msg){ msg.source = 'anton-comments'; try { parent.postMessage(msg, '*'); } catch (e) {} }
  function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]; }); }
  // ── Images (ENG-4410) ────────────────────────────────────────────────────
  // A comment names its images with marker lines the app writes
  // (![name](cowork-image:<id>)). This layer has no credential: the app reads
  // the images and pushes them down as Blobs ('images'), and takes the Files
  // pasted here with 'create' / 'reply', checking and storing them itself.
  var IMG_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
  var IMG_MAX = 5 * 1024 * 1024, IMG_COUNT = 4;
  var IMG_ID = /^[0-9a-f]{32}\.(?:png|jpg|gif|webp)$/;
  function imgMarks(){ return /!\[[^\]\n]*\]\(cowork-image:([0-9a-f]{32}\.(?:png|jpg|gif|webp))\)/g; }
  var imgUrls = {};  // id -> object URL of the Blob the app pushed
  function splitText(t){
    var ids = [], raw = String(t == null ? '' : t);
    var body = raw.replace(imgMarks(), function(_m, id){
      if (ids.indexOf(id) < 0) ids.push(id); return ''; });
    return {body: ids.length ? body.replace(/\n{3,}/g, '\n\n').trim() : raw, ids: ids};
  }
  function marksOf(t){ return (String(t || '').match(imgMarks()) || []).join('\n'); }
  function imgsHTML(ids){
    if (!ids.length) return '';
    return '<div class="act-imgs">' + ids.map(function(id){
      return '<span class="act-img" data-img="' + esc(id) + '">'
        + (imgUrls[id] ? '<img alt="" src="' + esc(imgUrls[id]) + '">' : '') + '</span>';
    }).join('') + '</div>';
  }
  function fillImages(){
    Array.prototype.forEach.call(document.querySelectorAll('.act-img[data-img]'), function(el){
      var url = imgUrls[el.getAttribute('data-img')];
      if (!url || el.firstChild) return;
      var im = document.createElement('img'); im.alt = ''; im.src = url; el.appendChild(im);
      clampPopV();
    });
  }
  document.addEventListener('click', function(e){
    var el = e.target && e.target.closest && e.target.closest('.act-img');
    if (el) { el.classList.toggle('act-big'); clampPopV(); }
  }, true);
  function imageFiles(list){
    return Array.prototype.filter.call(list || [], function(f){
      return f && IMG_TYPES.indexOf(String(f.type || '').toLowerCase()) >= 0
        && f.size > 0 && f.size <= IMG_MAX; });
  }

  function displayName(a){ a = String(a || ''); var i = a.indexOf('@');
    return i > 0 ? a.slice(0, i) : (a || 'Anonymous'); }
  function initials(a){
    var parts = displayName(a).split(/[^A-Za-z0-9]+/).filter(Boolean);
    if (!parts.length) return '?';
    return (parts[0].charAt(0) + (parts.length > 1 ? parts[1].charAt(0) : '')).toUpperCase();
  }
  var AVATAR_COLORS = ['#D99A1C','#1A8596','#5F8AD9','#C46FB0','#5FB87A','#D97A5F','#8F7FD9'];
  function avatarColor(a){ a = String(a || ''); var h = 0;
    for (var i = 0; i < a.length; i++) h = (h * 31 + a.charCodeAt(i)) >>> 0;
    return AVATAR_COLORS[h % AVATAR_COLORS.length]; }
  function avatar(a){ return '<span class="act-av" style="background:' + avatarColor(a)
    + '">' + esc(initials(a)) + '</span>'; }
  function viewerEmail(){ return (meViewer && meViewer.email) || ''; }
  var MONTHS = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'];
  function fmtDate(ts){ // "22 jun, 2026"
    if (!ts) return '';
    var d = new Date(ts * 1000);
    return d.getDate() + ' ' + MONTHS[d.getMonth()] + ', ' + d.getFullYear();
  }
  // Owner gating: the parent echoes the viewer with the list and the server
  // re-checks authorship on every PATCH/DELETE — this only hides affordances.
  function isMine(entry){
    return !!(meViewer && meViewer.user_id && entry && entry.author_user_id
      && String(entry.author_user_id) === String(meViewer.user_id));
  }
  function editedMark(at){
    if (!at) return '';
    var t = at; if (typeof t !== 'number') { t = Date.parse(t); t = isFinite(t) ? Math.floor(t/1000) : 0; }
    return ' <span class="act-edited" title="' + esc(fmtDate(t)) + '">(edited)</span>';
  }

  // The page background (body/html) is not a commentable element.
  function isPageBg(el){
    return el === document.body || el === document.documentElement;
  }
  // Gap-space hit test: margins/gaps hit-test to <body>, but they belong to
  // a container — descend to the deepest element whose bounding box contains
  // the pointer. If NO box contains it (gaps between siblings that are
  // direct body children), snap to the nearest element within 40px.
  // Null = genuine background.
  function elementAtPointByBox(cx, cy){
    var node = document.body, found = null;
    for (;;) {
      var next = null, best = null, bestD = 41, kids = node.children;
      for (var i = 0; i < kids.length; i++) {
        var k = kids[i];
        if (k.closest && k.closest('#act-mk,.act-pop,.act-menu,.act-hl')) continue;
        var r = k.getBoundingClientRect();
        if (!r.width || !r.height) continue;
        if (cx >= r.left && cx <= r.right && cy >= r.top && cy <= r.bottom) {
          next = k; break;
        }
        var dx = cx < r.left ? r.left - cx : (cx > r.right ? cx - r.right : 0);
        var dy = cy < r.top ? r.top - cy : (cy > r.bottom ? cy - r.bottom : 0);
        if (dx + dy < bestD) { bestD = dx + dy; best = k; }
      }
      if (next) { found = next; node = next; continue; }
      // Inside a container's own gap -> the container; otherwise nearest sibling.
      return found || best;
    }
  }
  function cssPath(el){
    if (!(el instanceof Element)) return null;
    // Page-level anchor (reached deliberately via Alt-expansion).
    if (isPageBg(el)) return 'body';
    if (el.id) return '#' + CSS.escape(el.id);
    var parts = [];
    while (el && el.nodeType === 1 && el !== document.body) {
      var sel = el.nodeName.toLowerCase();
      var parent = el.parentNode;
      if (!parent) break;
      if (el.id) { parts.unshift('#' + CSS.escape(el.id)); break; }
      var sibs = Array.prototype.filter.call(parent.children, function(c){
        return c.nodeName === el.nodeName; });
      if (sibs.length > 1) sel += ':nth-of-type(' + (sibs.indexOf(el) + 1) + ')';
      parts.unshift(sel);
      el = parent;
    }
    return parts.length ? parts.join(' > ') : null;
  }
  function rectOf(el){
    var r = el.getBoundingClientRect();
    return {top: r.top + window.scrollY, left: r.left + window.scrollX,
            right: r.right + window.scrollX, width: r.width, height: r.height};
  }

  function isClosed(c){ return c.status === 'resolved' || c.status === 'dismissed'; }

  // A resolvable but not-laid-out anchor (inactive slide/tab, collapsed
  // section) has zero client rects; isShown() distinguishes it from a truly
  // detached/removed element.
  function isShown(el){ return !!(el && el.getClientRects().length); }

  // ── Going to a comment's place (ENG-4383) ──────────────────────────────
  // A comment can sit on a slide, tab or section that isn't showing. The
  // page's own navigation is all there is to get there, so: an element
  // shown but moved out of view (a transform deck) counts as hidden too;
  // a control that points at a hidden ancestor (tab, accordion, #link) is
  // clicked; a closed <details> is opened; and failing those, the deck is
  // walked with the arrow keys, right then left, each step waiting for the
  // page to react and a direction ending when a key changes nothing.
  function inView(el){
    if (!isShown(el)) return false;
    var r = el.getBoundingClientRect();
    var w = document.documentElement.clientWidth, h = document.documentElement.clientHeight;
    // Off to the side is another slide; above or below is just a scroll away.
    return r.right > 0 && r.left < w && (r.width > 0 || r.height > 0) && h > 0;
  }
  function cssId(id){ return String(id).replace(/["\\]/g, '\\$&'); }
  function openControls(el){
    var det = el.closest && el.closest('details');
    while (det) { if (!det.open) det.open = true; det = det.parentElement && det.parentElement.closest('details'); }
    if (inView(el)) return true;
    for (var node = el; node && node !== document.body; node = node.parentElement) {
      if (!node.id) continue;
      var id = cssId(node.id);
      var ctl = document.querySelector('[aria-controls="' + id + '"],a[href="#' + id + '"],'
        + '[data-target="#' + id + '"],[data-bs-target="#' + id + '"],[data-tab="' + id + '"]');
      if (ctl && isShown(ctl) && !(ctl.closest && ctl.closest('.act-pop,#act-mk'))) {
        try { ctl.click(); } catch (e) {}
        if (inView(el)) return true;
      }
    }
    return false;
  }
  var WALK_MAX = 40, WALK_SETTLE_MS = 140;
  function pressKey(key){
    var code = key === 'ArrowRight' ? 39 : 37;
    var init = {key: key, code: key, keyCode: code, which: code, bubbles: true, cancelable: true};
    var target = document.activeElement && document.activeElement !== document.body ? document.activeElement : document.body;
    try { target.dispatchEvent(new KeyboardEvent('keydown', init)); } catch (e) {}
    try { target.dispatchEvent(new KeyboardEvent('keyup', init)); } catch (e) {}
  }
  // One step: press, then wait for the page to settle; `moved` is whether
  // anything outside this layer changed.
  function step(key, cb){
    var moved = false;
    var mo = new MutationObserver(function(list){
      for (var i = 0; i < list.length; i++) {
        var t = list[i].target;
        if (!(t.closest && t.closest('.act-pop,#act-mk,.act-hl'))) { moved = true; return; }
      }
    });
    mo.observe(document.body, {subtree: true, childList: true, attributes: true, characterData: true});
    pressKey(key);
    setTimeout(function(){ mo.disconnect(); cb(moved); }, WALK_SETTLE_MS);
  }
  function walk(el, key, left, done){
    if (inView(el)) { done(true); return; }
    if (left <= 0) { done(false); return; }
    step(key, function(moved){
      if (inView(el)) { done(true); return; }
      if (!moved) { done(false); return; }
      walk(el, key, left - 1, done);
    });
  }
  var revealing = false;
  function reveal(el, done){
    if (inView(el) || openControls(el)) { done(true); return; }
    if (revealing) { done(false); return; }
    revealing = true;
    var finish = function(found){ revealing = false; done(found); };
    walk(el, 'ArrowRight', WALK_MAX, function(found){
      if (found) { finish(true); return; }
      walk(el, 'ArrowLeft', WALK_MAX * 2, finish);
    });
  }

  // ── Markers ──────────────────────────────────────────────────────────────
  // One DOM node per live anchored thread, keyed by id and repositioned via
  // transform in a measure-then-write rAF batch (no per-marker reflow).
  var markerEls = {};   // id -> {el, target}
  var markersVisible = true;

  function markerHTML(c){
    var n = (c.replies || []).length + 1;
    return '<div class="act-pin' + (n > 9 ? ' act-2d' : '') + '">' + n + '</div>';
  }
  function syncMarkers(){
    var live = {};
    comments.forEach(function(c){
      if (!c.selector || isClosed(c)) return;
      var el; try { el = document.querySelector(c.selector); } catch (e) { el = null; }
      if (!el) return;                       // orphaned -> inbox-only ("unanchored")
      live[c.id] = {c: c, el: el};
    });
    // Drop markers whose thread is gone/closed/orphaned.
    Object.keys(markerEls).forEach(function(id){
      if (!live[id]) { markerEls[id].el.remove(); delete markerEls[id]; }
    });
    // Create/update the rest. New markers play a staggered entrance (20ms/pin)
    // so a batch reads as intentional, not as popping in.
    var born = 0;
    Object.keys(live).forEach(function(id){
      var c = live[id].c;
      var m = markerEls[id];
      if (!m) {
        var node = document.createElement('div');
        node.className = 'act-marker act-new';
        node.innerHTML = markerHTML(c);
        node.firstChild.style.animationDelay = (born++ * 20) + 'ms';
        node.addEventListener('click', function(ev){
          ev.stopPropagation(); openThread(c.id); });
        mk.appendChild(node);
        markerEls[id] = m = {el: node};
        setTimeout(function(){ node.classList.remove('act-new'); }, 500 + born * 20);
      } else {
        var pin = m.el.firstChild;
        var n = (c.replies || []).length + 1;
        if (pin.textContent !== String(n)) m.el.innerHTML = markerHTML(c);
      }
      m.target = live[id].el;
    });
    positionMarkers();
    reportAnchors();
  }
  // Measure phase then write phase — avoids interleaved layout thrash.
  function positionMarkers(){
    var ids = Object.keys(markerEls);
    var rects = ids.map(function(id){
      var t = markerEls[id].target;
      return t && t.isConnected ? t.getBoundingClientRect() : null;
    });
    // Clamp into the document box so pins never widen/lengthen the page
    // (a pin past the content edge would otherwise create scrollbars).
    var docW = document.documentElement.scrollWidth;
    var stacked = {};  // same anchor point -> cascade pins 5px down-right each
    ids.forEach(function(id, i){
      var m = markerEls[id], r = rects[i];
      // A display:none anchor (inactive carousel slide, collapsed section)
      // measures as a zero rect — hide its pin instead of stranding it at the
      // page origin; the Mutation/ResizeObserver re-run restores it when the
      // anchor becomes visible again.
      if (!r || (!r.width && !r.height)) { m.el.style.display = 'none'; return; }
      m.el.style.display = '';
      // Tail (bottom-left corner) sits at the element's top-right corner.
      var x = r.right + window.scrollX;
      var y = Math.max(0, r.top + window.scrollY - 28);
      var key = Math.round(x) + '_' + Math.round(y);
      var n = stacked[key] || 0;
      stacked[key] = n + 1;
      // Clamp AFTER the cascade offset so stacked pins can't recreate the
      // horizontal scrollbar the clamp exists to prevent.
      x = Math.min(x + n * 5, docW - 32);
      m.el.style.transform = 'translate(' + x + 'px,' + (y + n * 5) + 'px)';
    });
    if (!mk.classList.contains('act-ready')) {
      requestAnimationFrame(function(){ mk.classList.add('act-ready'); });
    }
  }
  var mkRAF = 0;
  function scheduleMarkers(){
    if (mkRAF) return;
    mkRAF = requestAnimationFrame(function(){ mkRAF = 0; positionMarkers(); });
  }
  var syncRAF = 0;
  function scheduleSync(){
    if (syncRAF) return;
    syncRAF = requestAnimationFrame(function(){ syncRAF = 0; syncMarkers(); });
  }
  // Classify anchored threads for the inbox: 'orphan' (selector no longer
  // resolves) or 'hidden' (resolves but not currently laid out — e.g. an
  // inactive slide/tab). 'anchored' is the norm and is NOT reported (absence
  // == anchored). isShown changes only via mutation/resize, never scroll, so
  // this is wired to those paths — not the scroll-bound positionMarkers.
  var _lastAnchorSig = '';
  function reportAnchors(){
    var states = {}, sig = [];
    comments.forEach(function(c){
      if (c.status === 'dismissed' || !c.selector) return;
      var el; try { el = document.querySelector(c.selector); } catch (e) { el = null; }
      var st = !el ? 'orphan' : (isShown(el) ? null : 'hidden');
      if (st) { states[c.id] = st; sig.push(c.id + ':' + st); }
    });
    sig = sig.sort().join('|');
    if (sig === _lastAnchorSig) return;   // signature-gate: silent on scroll / no-op mutations
    _lastAnchorSig = sig;
    send({type:'anchor-states', states: states});
  }
  var repRAF = 0;
  function scheduleReport(){
    if (repRAF) return;
    repRAF = requestAnimationFrame(function(){ repRAF = 0; reportAnchors(); });
  }
  // Keep positions honest across scrolls (incl. nested containers), resizes
  // and late layout shifts (images loading, DOM mutations).
  // scroll never changes isShown (only rect position) → reposition only.
  window.addEventListener('scroll', scheduleMarkers, {capture: true, passive: true});
  // resize / mutations / late layout CAN change visibility → also reclassify.
  window.addEventListener('resize', function(){ scheduleMarkers(); scheduleReport(); }, {passive: true});
  window.addEventListener('load', function(){ scheduleMarkers(); scheduleReport(); });
  if (document.fonts && document.fonts.ready)
    document.fonts.ready.then(function(){ scheduleMarkers(); scheduleReport(); });
  if (typeof ResizeObserver !== 'undefined')
    new ResizeObserver(function(){ scheduleMarkers(); scheduleReport(); }).observe(document.documentElement);
  if (typeof MutationObserver !== 'undefined')
    new MutationObserver(function(muts){
      for (var i = 0; i < muts.length; i++){
        var t = muts[i].target;
        // .act-hl included: its style writes on every mousemove would
        // otherwise trigger a full marker re-measure per frame.
        if (t === mk || (t.closest && t.closest('#act-mk,.act-pop,.act-menu,.act-hl')))
          continue;
        scheduleMarkers(); scheduleReport(); return;
      }
    }).observe(document.body, {childList: true, subtree: true, attributes: true,
                               attributeFilter: ['style', 'class', 'hidden']});

  // ── Popovers ─────────────────────────────────────────────────────────────
  var menuBtn = null, ghost = null, pendingCreate = false;
  function closeMenu(){
    if (menu) { menu.remove(); menu = null; }
    if (menuBtn) { menuBtn.classList.remove('act-on'); menuBtn = null; }
  }
  function setActivePin(id){
    mk.classList.toggle('act-focus', !!id);
    Object.keys(markerEls).forEach(function(k){
      markerEls[k].el.classList.toggle('act-active', k === id); });
  }
  function closePop(){
    // Closing the card any way keeps an edit in progress.
    if (editing) editing.commit();
    closeMenu();
    if (ghost) { ghost.remove(); ghost = null; }
    pendingCreate = false;
    setActivePin(null);
    if (pop) {
      // Quick exit fade — an entrance without an exit reads as broken.
      var dying = pop; pop = null;
      dying.style.pointerEvents = 'none';
      try {
        dying.animate([{opacity:1},{opacity:0,transform:'scale(.98)'}],
          {duration:120, easing:'ease-out'});
        setTimeout(function(){ dying.remove(); }, 120);
      } catch (e) { dying.remove(); }
    }
  }

  function popAt(left, top){
    closePop();
    pop = document.createElement('div');
    pop.className = 'act-pop';
    // Keys typed in the card stay in the card: a deck that moves on Space or
    // the arrows (and swallows them) never hears them, so a comment can have
    // spaces. Escape still reaches the layer, which closes the card with it.
    ['keydown', 'keypress', 'keyup'].forEach(function(type){
      pop.addEventListener(type, function(e){ if (e.key !== 'Escape') e.stopPropagation(); });
    });
    // Doc bounds measured BEFORE appending (the popover itself would inflate them).
    var docW = document.documentElement.scrollWidth;
    pop.__docH = document.documentElement.scrollHeight;
    pop.style.left = Math.max(8, Math.min(left, docW - 337)) + 'px';
    pop.style.top = Math.max(window.scrollY + 8, top) + 'px';
    document.body.appendChild(pop);
    return pop;
  }
  // Card sits to the right of the pin, tops aligned; flips to the left of the
  // pin when there's no room on the right.
  function popNextToMarker(x, y){
    var vwRight = window.scrollX + document.documentElement.clientWidth;
    var left = x + 35, flipped = false;
    if (left + 337 > vwRight) { left = x - 344; flipped = true; }
    var p = popAt(left, y);
    // Grow from the pin: origin follows which side of it the card sits on.
    p.style.transformOrigin = flipped ? 'top right' : 'top left';
    return p;
  }
  // Keep the card fully inside the VIEWPORT: cap its height to the visible
  // area (messages scroll internally), shift up off the bottom edge (toolbar
  // band included), never above the top. The user never scrolls to see it.
  function clampPopV(){
    if (!pop) return;
    var vTop = window.scrollY + 8;
    var vBottom = window.scrollY + document.documentElement.clientHeight - 88;
    pop.style.maxHeight = (vBottom - vTop) + 'px';
    var top = parseFloat(pop.style.top) || 0;
    var h = pop.offsetHeight;
    var limit = Math.min(pop.__docH - 8, vBottom);
    if (top + h > limit) top = limit - h;
    if (top < vTop) top = vTop;
    pop.style.top = top + 'px';
  }
  function headHTML(title, withThreadTools, resolved, showMenu){
    return '<div class="act-pop-head"><span class="act-pop-title">' + esc(title)
      + '</span><div class="act-pop-tools">'
      + (withThreadTools
        ? (showMenu
            ? '<button class="act-t-menu" aria-label="More actions" title="More">'
              + ICONS.dots + '</button>'
            : '')
          + '<button class="act-t-resolve' + (resolved ? ' act-resolved' : '')
          + '" aria-label="' + (resolved ? 'Reopen' : 'Mark as resolved')
          + '" title="' + (resolved ? 'Reopen' : 'Mark as resolved') + '">'
          + ICONS.check + '</button>'
        : '')
      + '<button class="act-t-close" aria-label="Close" title="Close">'
      + ICONS.x20 + '</button></div></div>';
  }
  function inputRowHTML(placeholder){
    return '<div class="act-reply">' + avatar(viewerEmail())
      + '<div class="act-inwrap"><textarea rows="1" placeholder="' + esc(placeholder)
      + '"></textarea><button class="act-send" title="Send">' + ICONS.arrowUp
      + '</button></div></div>';
  }
  // Pill grows into a rounded box as the text wraps. Height comes from a
  // hidden mirror with the same text metrics — real layout, immune to
  // textarea scrollHeight quirks.
  // `opts.images`: the input takes pasted or dropped images, handed to
  // onSubmit as its third argument (new comments and replies; not edits).
  function wireInput(root, onSubmit, opts){
    var wrap = root.querySelector('.act-inwrap');
    var ta = wrap.querySelector('textarea');
    var sendBtn = wrap.querySelector('.act-send');
    var pend = [], pendUrls = [];
    var strip = document.createElement('div');
    strip.className = 'act-pend';
    wrap.insertBefore(strip, ta);
    function hasContent(){ return !!ta.value.trim() || pend.length > 0; }
    function renderPend(){
      pendUrls.forEach(function(u){ URL.revokeObjectURL(u); });
      pendUrls = pend.map(function(f){ return URL.createObjectURL(f); });
      strip.innerHTML = pend.map(function(_f, i){
        return '<span class="act-pend-item"><img alt="" src="' + esc(pendUrls[i])
          + '"><button class="act-pend-x" data-i="' + i + '" aria-label="Remove image"'
          + ' title="Remove">\u00d7</button></span>';
      }).join('');
      Array.prototype.forEach.call(strip.querySelectorAll('.act-pend-x'), function(b){
        b.onclick = function(ev){ ev.stopPropagation();
          pend.splice(+b.getAttribute('data-i'), 1); renderPend(); grow(); ta.focus(); };
      });
    }
    function addFiles(list){
      imageFiles(list).forEach(function(f){ if (pend.length < IMG_COUNT) pend.push(f); });
      renderPend(); grow();
    }
    function submit(){
      if (!hasContent()) return;
      var images = pend.slice();
      onSubmit(ta.value.trim(), ta, images);
      pend = []; renderPend(); grow();
    }
    if (opts && opts.images) {
      ta.addEventListener('paste', function(e){
        var files = [];
        Array.prototype.forEach.call((e.clipboardData && e.clipboardData.items) || [], function(it){
          if (it.kind === 'file') { var f = it.getAsFile(); if (f) files.push(f); } });
        if (!imageFiles(files).length) return;
        e.preventDefault();
        addFiles(files);
      });
      wrap.addEventListener('dragover', function(e){
        if (e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], 'Files') >= 0)
          e.preventDefault(); });
      wrap.addEventListener('drop', function(e){
        if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
        e.preventDefault();
        addFiles(e.dataTransfer.files);
      });
    }
    var mir = document.createElement('div');
    mir.style.cssText = 'position:absolute!important;visibility:hidden!important;'
      + 'left:-9999px!important;top:0;white-space:pre-wrap!important;'
      + 'word-break:break-word!important;box-sizing:border-box!important;'
      + 'padding:0 4px!important;border:none!important;margin:0!important;'
      + 'font-family:inherit;font-size:14px;font-weight:400;line-height:24px;';
    wrap.appendChild(mir);  // removed with the popover
    function grow(){
      // Layout class FIRST (pill while empty, box once typing), THEN measure
      // at the textarea's live width so wraps count correctly.
      wrap.classList.toggle('act-multi', !!ta.value.length || pend.length > 0);
      sendBtn.classList.toggle('act-on', hasContent());
      mir.style.width = ta.clientWidth + 'px';
      mir.textContent = (ta.value || ' ') + '​';
      var h = Math.max(24, mir.offsetHeight);
      ta.style.height = Math.min(240, h) + 'px';
      // No scrollbar until the text actually exceeds the max height.
      ta.style.overflowY = h > 240 ? 'auto' : 'hidden';
      if (h > 240) ta.scrollTop = ta.scrollHeight;
      clampPopV();  // a growing input must not push the card off-viewport
    }
    ta.addEventListener('input', grow);
    ta.addEventListener('keydown', function(e){
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
    });
    sendBtn.addEventListener('click', submit);
    grow(); ta.focus();
  }
  // m = {author, created_at, text, edited_at, mine}; rid set for replies.
  // Own replies get hover edit/delete; the root comment's actions live in
  // the header "…" menu instead.
  // Own messages get a hover edit on their author line (ENG-4383): replies
  // edit and delete there; the root comment edits there too, and keeps
  // delete in the header's "…" menu with the thread's other actions.
  function msgHTML(m, rid){
    var acts = (m.mine && rid)
      ? '<span class="act-msg-acts">'
        + '<button class="act-m-edit" data-rid="' + esc(rid)
        + '" aria-label="Edit reply" title="Edit">' + ICONS.pencil + '</button>'
        + '<button class="act-m-del" data-rid="' + esc(rid)
        + '" aria-label="Delete reply" title="Delete">' + ICONS.trash + '</button></span>'
      : (m.mine && !rid)
        ? '<span class="act-msg-acts"><button class="act-m-edit-root"'
          + ' aria-label="Edit comment" title="Edit">' + ICONS.pencil + '</button></span>'
        : '';
    return '<div class="act-msg"' + (rid ? '' : ' data-root="1"') + '>'
      + '<div class="act-msg-head"><span class="act-msg-who">'
      + avatar(m.author) + '<span class="act-msg-name">' + esc(displayName(m.author))
      + '</span></span><span class="act-msg-date">' + esc(fmtDate(m.created_at))
      + editedMark(m.edited_at) + '</span>' + acts + '</div>'
      + '<div class="act-msg-text">' + esc(splitText(m.text).body) + '</div>'
      + imgsHTML(splitText(m.text).ids) + '</div>';
  }

  // Inline editor: hides the message text, mounts the growing input prefilled;
  // Enter saves (no-op edits skip the PATCH), Escape cancels.
  // The edit in progress, if any: `commit()` saves it (or drops a no-op),
  // so a click anywhere else keeps the change instead of losing it.
  var editing = null;

  // Inline editor: hides the message text, mounts the growing input prefilled.
  // Enter, the save button, or clicking away saves; Escape cancels; a no-op
  // edit skips the PATCH. Saving shows the new text in place and leaves the
  // card open (ENG-4383); the parent's next 'list' push confirms it.
  function startEdit(msgEl, oldText, onSave){
    if (!msgEl || msgEl.querySelector('.act-editbox')) return;
    if (editing) editing.commit();
    var textEl = msgEl.querySelector('.act-msg-text');
    textEl.style.display = 'none';
    var box = document.createElement('div');
    box.className = 'act-editbox';
    box.innerHTML = '<div class="act-inwrap"><textarea rows="1"></textarea>'
      + '<button class="act-send" title="Save">' + ICONS.arrowUp + '</button></div>';
    msgEl.appendChild(box);
    var ta = box.querySelector('textarea');
    // The images stay as they are: the box edits the words, and the image
    // markers go back after them on save.
    var words = splitText(oldText).body, marks = marksOf(oldText);
    ta.value = words;
    var done = false;
    function close(){
      done = true;
      if (editing && editing.box === box) editing = null;
      box.remove(); textEl.style.display = ''; clampPopV();
    }
    function save(text){
      if (done) return;
      text = String(text || '').trim();
      if (text === words || (!text && !marks)) { close(); return; }  // no-op: skip the PATCH
      var full = marks ? (text ? text + '\n\n' + marks : marks) : text;
      textEl.textContent = splitText(full).body;
      close();
      onSave(full);
    }
    editing = {box: box, commit: function(){ save(ta.value); }};
    wireInput(box, function(text){ save(text); });
    ta.dispatchEvent(new Event('input'));  // size to the prefilled text
    ta.addEventListener('keydown', function(e){
      if (e.key === 'Escape') { e.stopPropagation(); close(); }
    });
    // Focus leaving the box for anything outside it (another field in the
    // card, the page, the app) saves, as clicking away does.
    ta.addEventListener('blur', function(){
      setTimeout(function(){
        if (!done && !box.contains(document.activeElement)) save(ta.value);
      }, 0);
    });
  }

  // Mutations are posted to the parent, which owns the API creds; the next
  // 'list' push is the confirmation. The popover closes optimistically.
  function mutate(msg){ send(msg); closePop(); }

  // `at` (viewport coords) anchors an unanchored thread's card at that point;
  // the iframe fills the preview so viewport coords translate (+scroll) into
  // this document.
  function openThread(id, at){
    var c = comments.filter(function(x){ return x.id === id; })[0];
    if (!c) return;
    var m = markerEls[id];
    var p, orphan = false;
    if (m && m.target && isShown(m.target)) {
      var r = rectOf(m.target);
      p = popNextToMarker(r.right, r.top - 28);
    } else {
      orphan = true;
      if (at) {
        p = popAt(at.x + window.scrollX - 337, at.y + window.scrollY);
      } else {
        var cw = document.documentElement.clientWidth;
        p = popAt(window.scrollX + Math.max(8, (cw - 329) / 2), window.scrollY + 80);
      }
    }
    var resolved = c.status === 'resolved';
    var mine = isMine(c);
    var _el = null;
    if (c.selector) { try { _el = document.querySelector(c.selector); } catch (e) {} }
    var hiddenAnchor = !!(_el && !isShown(_el));
    var notice = (orphan && c.selector)
      ? (hiddenAnchor
          ? '<div class="act-orphan">' + ICONS.unlink
            + '<span>This comment is pinned to a part of the page that is not visible '
            + 'right now — open that slide or section to see it in place.</span></div>'
          : '<div class="act-orphan">' + ICONS.unlink
            + '<span>The annotated element was removed or changed, so this comment '
            + 'is no longer attached to the page.</span></div>')
      : '';
    var msgs = msgHTML({author: c.author, created_at: c.created_at, text: c.text,
                        edited_at: c.edited_at, mine: mine});
    (c.replies || []).forEach(function(rep){
      msgs += '<div class="act-div"></div>'
        + msgHTML({author: rep.author, created_at: rep.created_at, text: rep.text,
                   edited_at: rep.edited_at, mine: isMine(rep)}, rep.id);
    });
    p.innerHTML = headHTML('Comment', true, resolved, mine)
      + '<div class="act-pop-body">' + notice + '<div class="act-msgs">' + msgs + '</div>'
      + inputRowHTML('Reply') + '</div>';
    setActivePin(orphan ? null : id);
    p.querySelector('.act-t-close').onclick = closePop;
    p.querySelector('.act-t-resolve').onclick = function(){
      mutate({type:'status', id: id, status: resolved ? 'open' : 'resolved'}); };
    // "…" menu (author only): edit or delete the root comment.
    var tMenu = p.querySelector('.act-t-menu');
    if (tMenu) tMenu.onclick = function(ev){
      ev.stopPropagation();
      if (menu) { closeMenu(); return; }
      menu = document.createElement('div');
      menu.className = 'act-menu';
      menu.innerHTML =
        '<div class="act-menu-item" data-act="edit"><span class="act-mi-ic">'
        + ICONS.pencil + '</span><span>Edit</span></div>'
        + '<div class="act-menu-item" data-act="del"><span class="act-mi-ic">'
        + ICONS.trash + '</span><span>Delete</span></div>';
      // Right-aligned under the "…" button, which stays visibly pressed.
      menuBtn = this;
      menuBtn.classList.add('act-on');
      var br = this.getBoundingClientRect();
      menu.style.left = (br.right + window.scrollX - 145) + 'px';
      menu.style.top = (br.bottom + window.scrollY + 6) + 'px';
      menu.querySelector('[data-act="edit"]').onclick = function(){
        closeMenu();
        startEdit(p.querySelector('[data-root="1"]'), c.text,
          function(t){ c.text = t; send({type:'edit', id: id, text: t}); });
      };
      menu.querySelector('[data-act="del"]').onclick = function(){
        closeMenu();
        if (window.confirm('Delete this comment thread?'))
          mutate({type:'delete', id: id});
      };
      document.body.appendChild(menu);
    };
    // Hover edit on the root comment, beside its author and time.
    var rootEdit = p.querySelector('.act-m-edit-root');
    if (rootEdit) rootEdit.onclick = function(ev){
      ev.stopPropagation();
      startEdit(p.querySelector('[data-root="1"]'), c.text,
        function(t){ c.text = t; send({type:'edit', id: id, text: t}); });
    };
    // Hover actions on own replies.
    Array.prototype.forEach.call(p.querySelectorAll('.act-m-edit'), function(btn){
      btn.onclick = function(ev){
        ev.stopPropagation();
        var rid = btn.getAttribute('data-rid');
        var rep = (c.replies || []).filter(function(x){
          return String(x.id) === String(rid); })[0];
        if (!rep) return;
        startEdit(btn.closest('.act-msg'), rep.text,
          function(t){ rep.text = t; send({type:'edit-reply', id: id, replyId: rid, text: t}); });
      };
    });
    Array.prototype.forEach.call(p.querySelectorAll('.act-m-del'), function(btn){
      btn.onclick = function(ev){
        ev.stopPropagation();
        var rid = btn.getAttribute('data-rid');
        if (window.confirm('Delete this reply?'))
          mutate({type:'delete-reply', id: id, replyId: rid});
      };
    });
    wireInput(p, function(text, ta, images){
      // Append in place — no close/reopen flash of the popover. The parent's
      // next 'list' push reconciles ids/timestamps (and shows the images).
      send({type:'reply', id: id, text: text, images: images});
      var box = p.querySelector('.act-msgs');
      box.insertAdjacentHTML('beforeend', '<div class="act-div"></div>'
        + msgHTML({author: viewerEmail(),
                   created_at: Math.floor(Date.now() / 1000), text: text}));
      box.scrollTop = box.scrollHeight;
      ta.value = '';
      ta.dispatchEvent(new Event('input'));
      ta.focus();
    }, {images: true});
    var msgsEl = p.querySelector('.act-msgs');
    msgsEl.scrollTop = msgsEl.scrollHeight;
    clampPopV();
  }

  function openAdd(el){
    var r = rectOf(el);
    // Open the popover FIRST (its closePop() clears any previous ghost),
    // THEN drop the ghost pin at the anchor (pin + floating pill).
    var p = popNextToMarker(r.right, r.top - 28);
    var docW = document.documentElement.scrollWidth;
    ghost = document.createElement('div');
    ghost.className = 'act-marker act-new';
    ghost.style.pointerEvents = 'none';
    ghost.style.zIndex = '2147483641';
    ghost.innerHTML = '<div class="act-pin"></div>';
    ghost.style.transform = 'translate('
      + Math.min(r.right + window.scrollX, docW - 32) + 'px,'
      + Math.max(0, r.top + window.scrollY - 28) + 'px)';
    document.body.appendChild(ghost);
    p.classList.add('act-compose');
    p.innerHTML = '<div class="act-inwrap"><textarea rows="1" '
      + 'placeholder="Add a comment"></textarea>'
      + '<button class="act-send" title="Send">' + ICONS.arrowUp + '</button></div>';
    wireInput(p, function(text, ta, images){
      ta.disabled = true;
      // Composer goes away immediately; the ghost pin stays with a spinner
      // until the parent confirms via the next 'list' push (then the real
      // pin replaces it).
      if (pop) { pop.remove(); pop = null; }
      if (ghost) ghost.firstChild.innerHTML = '<span class="act-spin"></span>';
      pendingCreate = true;
      send({type:'create', selector: cssPath(el), text: text, images: images});
      // Failure fallback: if no list push arrives (create failed upstream),
      // don't leave a spinner pinned forever. Images upload first, so wait longer.
      setTimeout(function(){ if (pendingCreate) closePop(); }, images.length ? 30000 : 8000);
    }, {images: true});
    clampPopV();
  }

  // ── Comment mode (hover highlight + place) ───────────────────────────────
  function setMode(on){
    mode = on;
    document.body.classList.toggle('act-mode', on);
    if (!on) hl.style.opacity = '0';
    send({type:'mode', active: on});
  }

  document.addEventListener('mousemove', function(e){
    if (!mode) return;
    var el = e.target;
    if (!el || el === hl) return;
    hoverExpanded = false;  // moving the mouse resets any Alt-expansion
    if (el.closest && el.closest('#act-mk,.act-pop,.act-menu')) {
      hoverEl = null; hl.style.opacity = '0'; return; }
    // In a gap? Container whose box holds the pointer, else nearest sibling,
    // else the page itself (body = the container of everything).
    if (isPageBg(el))
      el = elementAtPointByBox(e.clientX, e.clientY) || document.body;
    hoverEl = el;
    var r = rectOf(el);
    hl.style.opacity = '1';
    hl.style.top = r.top + 'px'; hl.style.left = r.left + 'px';
    hl.style.width = r.width + 'px'; hl.style.height = r.height + 'px';
  }, true);

  document.addEventListener('click', function(e){
    if (!mode) return;
    if (e.target.closest && e.target.closest('#act-mk,.act-pop,.act-menu')) return;
    e.preventDefault(); e.stopPropagation();
    // A click away from an edit in progress saves it; it doesn't start a
    // new comment as well.
    if (editing) { editing.commit(); return; }
    var el = hoverEl || e.target;
    // Gap clicks resolve like hover: containing box -> nearest sibling ->
    // the page itself. The full-page highlight makes a body pick obvious.
    if (el && isPageBg(el) && !hoverExpanded)
      el = elementAtPointByBox(e.clientX, e.clientY) || document.body;
    if (!el) { closePop(); return; }
    openAdd(el);
  }, true);

  // Hold/press Alt (⌥) while hovering to expand the selection to the
  // CONTAINING element — each press steps one level up the tree, all the way
  // to the page body; moving the mouse resets to the element under the
  // pointer. Click anchors whatever is highlighted.
  var hoverExpanded = false;
  document.addEventListener('keydown', function(e){
    if (!mode || e.key !== 'Alt' || !hoverEl) return;
    var up = hoverEl.parentElement;
    if (!up || up === document.documentElement) return;
    e.preventDefault();
    hoverEl = up;
    hoverExpanded = true;
    var r = rectOf(up);
    hl.style.opacity = '1';
    hl.style.top = r.top + 'px'; hl.style.left = r.left + 'px';
    hl.style.width = r.width + 'px'; hl.style.height = r.height + 'px';
  });

  // Events born inside our UI never reach the artifact's own document-level
  // handlers (e.g. a page's "click outside" logic).
  ['mousedown','pointerdown','click'].forEach(function(t){
    document.body.addEventListener(t, function(e){
      if (e.target.closest && e.target.closest('.act-pop,.act-menu,#act-mk'))
        e.stopPropagation();
    });
  });

  // Menu closes on ANY outside click — capture phase, so clicks inside the
  // popover (which stop propagation at body) still dismiss it.
  document.addEventListener('click', function(e){
    if (!menu) return;
    var t = e.target;
    if (t.closest && (t.closest('.act-menu') || t.closest('.act-t-menu'))) return;
    closeMenu();
  }, true);

  // Outside click closes the popover, unless a comment is being edited
  // there: then the click saves the edit and the card stays open.
  document.addEventListener('click', function(e){
    if (!pop) return;
    var t = e.target;
    if (t.closest && (t.closest('.act-pop') || t.closest('#act-mk'))) return;
    if (editing) { editing.commit(); return; }
    closePop();
  });

  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape') { if (menu) closeMenu(); else if (pop) closePop(); return; }
    // Forward the C shortcut when not typing (parent owns the toolbar state).
    if ((e.key === 'c' || e.key === 'C') && !e.metaKey && !e.ctrlKey && !e.altKey) {
      var t = e.target;
      if (t && (t.nodeName === 'INPUT' || t.nodeName === 'TEXTAREA' || t.isContentEditable))
        return;
      send({type:'shortcut', key:'c'});
    }
  });

  window.addEventListener('message', function(ev){
    var d = ev.data || {};
    if (d.source !== 'anton-comments') return;
    if (d.type === 'list') {
      comments = Array.isArray(d.comments) ? d.comments : [];
      if ('viewer' in d) meViewer = d.viewer || null;
      if ('markersVisible' in d) {
        markersVisible = !!d.markersVisible;
        mk.classList.toggle('act-off', !markersVisible);
      }
      // A pending create is confirmed by the parent's next push: the real
      // pin exists in this set, so the ghost/spinner can go.
      if (pendingCreate) closePop();
      scheduleSync();
      send({type:'count', count: comments.length});
    }
    else if (d.type === 'images') {
      // Only the app's own pushes: a script in the page posting as the app
      // must not plant pictures in someone's comment.
      if (ev.source !== parent) return;
      var given = d.images || {};
      Object.keys(given).forEach(function(id){
        var blob = given[id];
        if (!IMG_ID.test(id) || imgUrls[id] || !(blob instanceof Blob)
            || IMG_TYPES.indexOf(String(blob.type || '').toLowerCase()) < 0) return;
        imgUrls[id] = URL.createObjectURL(blob);
      });
      fillImages();
    }
    else if (d.type === 'markers') {
      markersVisible = !!d.visible;
      mk.classList.toggle('act-off', !markersVisible);
      if (!markersVisible) closePop();
    }
    else if (d.type === 'enter-mode') setMode(true);
    else if (d.type === 'exit-mode') { setMode(false); closePop(); }
    // Any other UI action (inbox, hide, …) discards an in-progress popover.
    else if (d.type === 'close-pop') closePop();
    else if (d.type === 'focus') {
      var c = comments.filter(function(x){ return x.id === d.commentId; })[0];
      if (!c) return;
      var el = null;
      if (c.selector) { try { el = document.querySelector(c.selector); } catch (e) {} }
      if (!el) { openThread(c.id, d.at || null); return; }  // unanchored / orphaned: notice
      // Take the page to the comment first (another slide, a closed tab or
      // section), then open it there; if it can't be reached, the card
      // opens centred with the "not visible right now" notice.
      var focusId = c.id, at = d.at || null;
      reveal(el, function(found){
        if (!found) { openThread(focusId, at); return; }
        el.scrollIntoView({behavior:'smooth', block:'center'});
        setTimeout(function(){ openThread(focusId); }, 250);
      });
    }
    else if (d.type === 'hl-on') {
      if (mode) return;
      var hc = comments.filter(function(x){ return x.id === d.commentId; })[0];
      var hel = null;
      if (hc && hc.selector) { try { hel = document.querySelector(hc.selector); } catch (e) {} }
      if (!hel || !isShown(hel)) { hl.style.opacity = '0'; return; }
      var hr = rectOf(hel);
      hl.style.opacity = '1';
      hl.style.top = hr.top + 'px'; hl.style.left = hr.left + 'px';
      hl.style.width = hr.width + 'px'; hl.style.height = hr.height + 'px';
    }
    else if (d.type === 'hl-off') { if (!mode) hl.style.opacity = '0'; }
  });

  // Announce readiness so the parent pushes the current comment list.
  send({type:'ready'});
})();
"""

LAYER_JS = (LAYER_JS
            .replace("__ICONS__", _ICONS_JS)
            .replace("__LAYER_CSS__", json.dumps(_LAYER_CSS)))


def script_tag(js: str) -> str:
    """Wrap ``js`` in a ``<script>`` element safe to splice into a document.

    A literal ``</script>`` in the payload would break out of the injected tag,
    so it is escaped. Shared with preview_shim: the escaping rule is a
    boundary, and both injected scripts must apply the same one.
    """
    return "<script>%s</script>" % js.replace("</script>", "<\\/script>")


# Precomputed once: the JS is a constant blob.
_SCRIPT_TAG = script_tag(LAYER_JS)
# Case-insensitive so we anchor to the real body close in mixed-case documents,
# without allocating a full lowercased copy of the (possibly large) HTML.
_BODY_CLOSE_RE = re.compile(r"</body\s*>", re.IGNORECASE)


def inject_layer(html: str) -> str:
    """Append the comment marker layer to an artifact's HTML document.

    Inserts before the last ``</body>`` when present (so the layer's absolutely
    positioned nodes live inside the body), else falls back to appending.
    """
    matches = list(_BODY_CLOSE_RE.finditer(html))
    if matches:
        idx = matches[-1].start()
        return html[:idx] + _SCRIPT_TAG + html[idx:]
    return html + _SCRIPT_TAG
