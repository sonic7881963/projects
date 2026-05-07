"use strict";
cc._RF.push(module, 'ab722XkV0BFB5hi6AcvAkW1', 'left - 001');
// 3.16小游戏/command_TypeScript/level5/left - 001.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var global___001_1 = require("./global - 001");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var mainCharacter = /** @class */ (function (_super) {
    __extends(mainCharacter, _super);
    function mainCharacter() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    mainCharacter.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    mainCharacter.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___001_1.default.attack == true) {
            global___001_1.default.minion1_hp -= 17;
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        global___001_1.default.attack = false;
    };
    mainCharacter.prototype.onCollisionStay = function (other) {
        global___001_1.default.attack = false;
        global___001_1.default.minion_attack = false;
    };
    mainCharacter.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        if (global___001_1.default.minion1_hp <= 0) {
            other.node.active = false;
            global___001_1.default.attack = false;
            global___001_1.default.minion_attack = false;
            global___001_1.default.main_hp = 163;
            global___001_1.default.minion1_hp = 163;
            global___001_1.default.minion2_hp = 163;
            global___001_1.default.minion3_hp = 163;
            cc.director.loadScene("fight6");
        }
        else {
            other.node.children[0].setContentSize(global___001_1.default.minion1_hp, 19);
        }
    };
    mainCharacter.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    mainCharacter.prototype.update = function (dt) {
        if (global___001_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], mainCharacter.prototype, "label", void 0);
    __decorate([
        property
    ], mainCharacter.prototype, "text", void 0);
    mainCharacter = __decorate([
        ccclass
    ], mainCharacter);
    return mainCharacter;
}(cc.Component));
exports.default = mainCharacter;

cc._RF.pop();