-- Native Linux controller for ToriLLE.
-- LuaSocket cannot be loaded by the native Steam client, so this script uses
-- atomic files in .torille at the game root.

local BRIDGE = ".torille/"
local SETTINGS = BRIDGE .. "settings"
local STATE = BRIDGE .. "state"
local ACTION = BRIDGE .. "action"
local RESET = BRIDGE .. "reset"

local NUM_JOINTS = 20
local NUM_LIMBS = 21

-- Training mode runs as fast as possible, watching mode limits the frame rate.
local OPTIONS_NO_RENDERING = { fixedframerate = 0 }
local OPTIONS_RENDERING = { fixedframerate = 1 }

local drawing = nil
local first_game = true
local waiting_for_action = false
local waiting_for_reset = false
local replay_file = "None"
local game_mod = "classic"
-- Milliseconds to keep checking for an action inside the freeze hook before
-- falling back to the per-frame poll (saves up to one frame of latency).
local spin_ms = 0

local function read_file(path)
    local file = io.open(path, "r", true)
    if file == nil then return nil end
    local content = file:read("*a")
    file:close()
    -- Consumed files stay behind empty, which is not a message.
    if content == "" then return nil end
    local cleared = io.open(path, "w", true)
    if cleared ~= nil then cleared:close() end
    return content
end

local function write_file(path, content)
    local file = io.open(path, "w", true)
    if file == nil then
        echo("ToriLLE cannot write " .. path)
        return
    end
    file:write(content)
    file:close()
end

local function split_numbers(content)
    local values = {}
    for value in string.gmatch(content, "([^,]+)") do
        table.insert(values, tonumber(value))
    end
    return values
end

local function split_strings(content)
    local values = {}
    for value in string.gmatch(content, "([^,]+)") do
        table.insert(values, value)
    end
    return values
end

local function build_state()
    local state = {}
    for player = 0, 1 do
        for body = 0, NUM_LIMBS - 1 do
            local info = get_body_info(player, body)
            table.insert(state, info.pos.x)
            table.insert(state, info.pos.y)
            table.insert(state, info.pos.z)
        end
        for body = 0, NUM_LIMBS - 1 do
            local x, y, z = get_body_linear_vel(player, body)
            table.insert(state, x)
            table.insert(state, y)
            table.insert(state, z)
        end
        local rotation = get_body_info(player, 4).rot
        for index = 0, 15 do table.insert(state, rotation["r" .. index]) end
        for joint = 0, NUM_JOINTS - 1 do
            table.insert(state, get_joint_info(player, joint).state)
        end
        table.insert(state, get_grip_info(player, 12))
        table.insert(state, get_grip_info(player, 11))
        table.insert(state, get_player_info(player).injury)
    end
    local world = get_world_state()
    table.insert(state, world.selected_player)
    table.insert(state, world.game_frame)
    table.insert(state, world.match_frame)
    table.insert(state, world.match_turn_frame)
    return table.concat(state, ",")
end

local function apply_render_options(draw_game)
    local options = draw_game and OPTIONS_RENDERING or OPTIONS_NO_RENDERING
    for option, value in pairs(options) do
        set_option(option, value)
    end
end

local function apply_settings()
    local content = read_file(SETTINGS)
    if content == nil then return false end
    local draw_line, settings_line, spin_line = string.match(content, "([^\n]*)\n([^\n]*)\n?([^\n]*)")
    if settings_line == nil then return false end
    spin_ms = tonumber(spin_line) or 0
    local draw_game = draw_line == "1"
    if draw_game ~= drawing then
        drawing = draw_game
        apply_render_options(draw_game)
    end
    local settings = split_strings(settings_line)
    if settings[21] ~= game_mod then
        game_mod = settings[21]
        run_cmd("set mod " .. game_mod)
    end
    if settings[1] ~= "0" then
        run_cmd("set matchframes " .. settings[2])
        run_cmd("set turnframes " .. settings[3])
        run_cmd("set engagedistance " .. settings[4])
        run_cmd("set engageheight " .. settings[5])
        run_cmd("set engagerotation " .. settings[6])
        run_cmd("set gravity " .. settings[7] .. " " .. settings[8] .. " " .. settings[9])
        run_cmd("set damage " .. settings[10])
        run_cmd("set dismemberment " .. settings[11])
        run_cmd("set dismemberthreshold " .. settings[12])
        run_cmd("set fracture " .. settings[13])
        run_cmd("set fracturethreshold " .. settings[14])
        run_cmd("set disqualification " .. settings[15])
        run_cmd("set dqflag " .. settings[16])
        run_cmd("set dqtimeout " .. settings[17])
        run_cmd("set dojotype " .. settings[18])
        run_cmd("set dojosize " .. settings[19])
    end
    replay_file = settings[20]
    return true
end

local function make_move(actions)
    local offset = 1
    for joint = 0, NUM_JOINTS - 1 do
        set_joint_state(0, joint, actions[joint + offset])
    end
    set_grip_info(0, BODYPARTS.L_HAND, actions[NUM_JOINTS + offset])
    set_grip_info(0, BODYPARTS.R_HAND, actions[NUM_JOINTS + 1 + offset])
    offset = NUM_JOINTS + 3
    for joint = 0, NUM_JOINTS - 1 do
        set_joint_state(1, joint, actions[joint + offset])
    end
    set_grip_info(1, BODYPARTS.L_HAND, actions[NUM_JOINTS + offset])
    set_grip_info(1, BODYPARTS.R_HAND, actions[NUM_JOINTS + 1 + offset])
end

local function take_action()
    local action = read_file(ACTION)
    if action == nil then return false end
    make_move(split_numbers(action))
    waiting_for_action = false
    step_game()
    return true
end

local function request_action(spin)
    if waiting_for_action or waiting_for_reset then return end
    write_file(STATE, build_state() .. "\n")
    waiting_for_action = true
    if spin and spin_ms > 0 then
        local deadline = os.clock() + spin_ms / 1000
        while os.clock() < deadline do
            if take_action() then return end
        end
    end
end

local function poll_bridge()
    if waiting_for_action then
        take_action()
    elseif waiting_for_reset and read_file(RESET) ~= nil then
        if apply_settings() then
            waiting_for_reset = false
            run_cmd("reset")
        end
    end
end

local function finish_game()
    if replay_file ~= "None" then run_cmd("savereplay " .. replay_file) end
    local winner = get_world_state().winner + 1
    write_file(STATE, "end:" .. winner .. "," .. build_state() .. "\n")
    waiting_for_action = false
    waiting_for_reset = true
end

local function start_game()
    if first_game then
        first_game = false
        if apply_settings() then
            add_hook("end_game", "torille_native", finish_game)
            run_cmd("reset")
        end
    else
        -- Do not spin here: stepping from the new_game hook advances a
        -- different number of frames than stepping from pre_draw.
        request_action(false)
    end
end

close_menu()
add_hook("new_game", "torille_native", start_game)
add_hook("enter_freeze", "torille_native", function() request_action(true) end)
add_hook("pre_draw", "torille_native", poll_bridge)
run_cmd("loadmod classic")
